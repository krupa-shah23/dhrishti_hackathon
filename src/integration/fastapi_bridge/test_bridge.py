"""
Run manually against a running backend mock (see below) or a real backend.
Start the bridge: uvicorn main:app --reload --port 8000
Start a mock backend on :5000 that just echoes 200 OK, or point BACKEND_URL at real one.
"""
import asyncio
import time
import subprocess
import signal
import httpx

BRIDGE_URL = "http://localhost:8000"


async def test_process_triggers_incremental_posts():
    """Confirms /process returns immediately (non-blocking) and events post async after."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        resp = await client.post(f"{BRIDGE_URL}/process", json={"video_id": "clip_test1", "path": "/fake/path.mp4"})
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 1.0, "process() should return immediately, not block on pipeline"
        print(f"/process returned in {elapsed:.2f}s — OK (non-blocking)")

        await asyncio.sleep(3)  # let mock pipeline run
        debug = await client.get(f"{BRIDGE_URL}/_debug/posted_events")
        count = debug.json()["count"]
        assert count > 0, "expected some events to have posted by now"
        print(f"{count} events posted incrementally — OK")


def test_mid_run_kill_survives():
    """
    Starts the bridge as a subprocess, triggers /process, kills the process
    mid-run, and confirms events already posted to the backend before the kill
    are NOT lost (backend should show partial events, not zero).
    Run this test file directly: python test_bridge.py
    """
    proc = subprocess.Popen(
        ["uvicorn", "main:app", "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(2)  # let server boot

    try:
        resp = httpx.post(f"{BRIDGE_URL}/process", json={"video_id": "clip_kill_test", "path": "/fake.mp4"})
        assert resp.status_code == 200
        time.sleep(0.5)  # let 1-2 events post before kill
        proc.send_signal(signal.SIGKILL)
        print("Killed bridge mid-run. Now check backend for clip_kill_test — "
              "expect >0 but <5 events (mock generates 5), confirming partial POSTs survived.")
    finally:
        proc.wait(timeout=5)


def test_no_duplicate_on_retry():
    """
    Manually re-POST the same event twice to /internal/events on the backend
    directly (simulating a bridge retry) and confirm the backend shows exactly
    one record for that event_id, not two. Requires backend running.
    """
    import uuid
    event = {
        "event_id": str(uuid.uuid4()),
        "video_id": "clip_dup_test",
        "start": 0.0, "end": 5.0,
        "seat_id": None, "event_type": "test",
        "object_detected": False, "object_confidence": None,
        "notes": "dup test"
    }
    import os
    backend = os.environ.get("BACKEND_URL", "http://localhost:5000")
    r1 = httpx.post(f"{backend}/internal/events", json=event)
    r2 = httpx.post(f"{backend}/internal/events", json=event)  # retry
    print(f"First POST: {r1.status_code}, Retry POST: {r2.status_code}")
    print("Manually confirm backend DB has exactly ONE record for this event_id.")


if __name__ == "__main__":
    asyncio.run(test_process_triggers_incremental_posts())
    test_mid_run_kill_survives()
    test_no_duplicate_on_retry()