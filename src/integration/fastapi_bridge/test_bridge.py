"""
End-to-end integration tests for FastAPI Bridge and Backend webhook endpoints.
"""
import os
import sys
import time
import asyncio
import uuid
import subprocess
from pathlib import Path
import httpx

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8000")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


def _wait_for_server(url: str, timeout: float = 25.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


async def _reset_backend():
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BACKEND_URL}/_debug/reset")
        assert resp.status_code == 200, f"Failed to reset backend: {resp.status_code}"


async def test_process_triggers_incremental_posts():
    """Confirms /process returns immediately (non-blocking) and events post async after."""
    await _reset_backend()

    bridge_dir = Path(__file__).parent.resolve()
    repo_root = bridge_dir.parents[2].resolve()
    env = os.environ.copy()
    env["USE_MOCK"] = "true"
    env["BACKEND_URL"] = BACKEND_URL
    env["PYTHONPATH"] = str(repo_root)

    bridge_proc = None
    try:
        r = httpx.get(f"{BRIDGE_URL}/health", timeout=0.5)
        running = (r.status_code == 200)
    except Exception:
        running = False

    if not running:
        bridge_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(bridge_dir),
            env=env,
        )
        assert _wait_for_server(BRIDGE_URL, timeout=25.0), "Bridge server failed to start on port 8000"

    try:
        async with httpx.AsyncClient() as client:
            start = time.time()
            resp = await client.post(
                f"{BRIDGE_URL}/process",
                json={"video_id": "clip_test1", "path": "/fake/path.mp4"},
            )
            elapsed = time.time() - start
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            assert elapsed < 1.0, "process() should return immediately, not block on pipeline"
            print(f"/process returned in {elapsed:.2f}s — OK (non-blocking)")

            await asyncio.sleep(3)  # let mock pipeline run
            debug = await client.get(f"{BACKEND_URL}/_debug/events?video_id=clip_test1")
            assert debug.status_code == 200
            events = debug.json().get("events", [])
            assert len(events) > 0, "expected some events to have posted by now"
            print(f"{len(events)} events posted incrementally to backend — OK")
    finally:
        if bridge_proc is not None:
            bridge_proc.kill()
            bridge_proc.wait(timeout=5)


def test_mid_run_kill_survives():
    """
    Starts the bridge as a subprocess, triggers /process, kills the process
    mid-run, and confirms events already posted to the backend before the kill
    are NOT lost (backend should show partial events, not zero).
    """
    asyncio.run(_reset_backend())

    bridge_dir = Path(__file__).parent.resolve()
    repo_root = bridge_dir.parents[2].resolve()
    env = os.environ.copy()
    env["USE_MOCK"] = "true"
    env["BACKEND_URL"] = BACKEND_URL
    env["PYTHONPATH"] = str(repo_root)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(bridge_dir),
        env=env,
    )
    assert _wait_for_server(BRIDGE_URL, timeout=25.0), "Bridge server failed to start on port 8000"

    try:
        resp = httpx.post(
            f"{BRIDGE_URL}/process",
            json={"video_id": "clip_kill_test", "path": "/fake.mp4"},
        )
        assert resp.status_code == 200
        time.sleep(0.25)  # let 1-3 events post before kill (mock generates 5 with 0.1s delay)
        proc.kill()
        print("Killed bridge mid-run.")
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait(timeout=5)

    events_resp = httpx.get(f"{BACKEND_URL}/_debug/events?video_id=clip_kill_test")
    assert events_resp.status_code == 200
    surviving = events_resp.json().get("events", [])
    assert 0 < len(surviving) < 5, (
        f"Expected 0 < len(surviving) < 5, got {len(surviving)}"
    )
    print(f"Mid-run kill survived: {len(surviving)} events in backend (expected >0 and <5) — OK")

    comp_resp = httpx.get(f"{BACKEND_URL}/_debug/complete/clip_kill_test")
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data.get("status") != "done", (
        f"Expected status != 'done', got {comp_data.get('status')}"
    )
    print(f"Completion status is '{comp_data.get('status')}' != 'done' — OK")


def test_no_duplicate_on_retry():
    """
    Manually re-POST the same event twice to /internal/events on the backend
    directly (simulating a bridge retry) and confirm the backend shows exactly
    one record for that event_id, not two.
    """
    asyncio.run(_reset_backend())

    event_id = str(uuid.uuid4())
    event = {
        "event_id": event_id,
        "video_id": "clip_dup_test",
        "camera_id": "cam_clip_dup_test",
        "start": 0.0,
        "end": 5.0,
        "duration": 5.0,
        "seat_id": None,
        "event_type": "test",
        "object_detected": False,
        "object_confidence": None,
        "person_ids": ["person_1"],
        "activities": ["test"],
        "avg_motion_intensity": 10.0,
        "peak_intensity": 25.0,
        "mog2_foreground_ratio": 0.05,
        "repetition_count": 0,
        "invigilator_excluded": False,
        "exam_phase": "main_exam",
        "severity_score": 0.2,
        "explanation": "Test explanation",
        "confidence": 0.8,
        "color_tag": "yellow",
        "notes": "dup test",
    }
    r1 = httpx.post(f"{BACKEND_URL}/internal/events", json=event)
    assert r1.status_code == 200
    assert r1.json().get("is_new") is True, f"Expected is_new=True on first POST, got {r1.json()}"

    r2 = httpx.post(f"{BACKEND_URL}/internal/events", json=event)  # retry
    assert r2.status_code == 200
    assert r2.json().get("is_new") is False, f"Expected is_new=False on duplicate POST, got {r2.json()}"

    resp = httpx.get(f"{BACKEND_URL}/_debug/events?video_id=clip_dup_test")
    assert resp.status_code == 200
    events = resp.json().get("events", [])
    matching = [e for e in events if e.get("event_id") == event_id]
    assert len(matching) == 1, f"Expected exactly 1 matching record for {event_id}, got {len(matching)}"
    print(f"First POST: is_new={r1.json().get('is_new')}, Retry POST: is_new={r2.json().get('is_new')} — OK (dedup verified)")


if __name__ == "__main__":
    asyncio.run(test_process_triggers_incremental_posts())
    test_mid_run_kill_survives()
    test_no_duplicate_on_retry()