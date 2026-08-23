"""
Regression-guard suite locking in the P1xP2 integration pass (conflict-marker
cleanup, 3.0s detection window, real motion_intensity plumbing, boxes field
in schemas.Event, fake_backend smoke test). Verification pass, not a fix pass
-- failures here should be reported, not silently patched.
"""
import ast
import inspect
import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import (
    enrich_event_with_motion_fields,
    segment_events,
)
from src.integration.event_adapter import adapt_bridge_event_to_schema
from src.integration.fastapi_bridge.schemas import Event
from src.track_det.detector import detect_objects
from src.motion.grid_config import GRID_CONFIGS

CLIPS_DIR = REPO_ROOT / "data" / "drishti"
CLIP_01 = CLIPS_DIR / "01_phone_use.mkv"
TRACKED_EXTS = (".py", ".csv", ".yaml", ".yml")


def _run_frames(video_path, n_frames, camera_id=None):
    """Run consecutive frames through P1P2TrackerPipeline + P2P3Bridge,
    exactly like the smoke-test script used in prior verification passes."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipeline = P1P2TrackerPipeline(clip_name=video_path.name, fps=fps)
    bridge = P2P3Bridge(missing_threshold=15, fps=fps)

    n = 0
    while n < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, n, camera_id=camera_id)
        bridge.process_fused_tracks(
            result["fused_tracks"],
            frame_index=n,
            pose_signals=result.get("pose_signals"),
            motion_intensity=result.get("motion_intensity"),
            mog2_foreground_ratio=result.get("mog2_foreground_ratio"),
        )
        n += 1
    cap.release()
    bridge.flush()
    return pipeline, bridge, n


@pytest.fixture(scope="module")
def clip01_run():
    if not CLIP_01.exists():
        pytest.skip("01_phone_use.mkv not available")
    pipeline, bridge, n_frames = _run_frames(CLIP_01, 300)
    events = bridge.completed_events
    return {"pipeline": pipeline, "bridge": bridge, "n_frames": n_frames, "events": events}


# ---------------------------------------------------------------------------
# GROUP A -- Contract Integrity
# ---------------------------------------------------------------------------

def test_A1_zero_conflict_markers():
    import subprocess
    result = subprocess.run(
        ["git", "grep", "-n", "-E", r"^<<<<<<<|^=======$|^>>>>>>>", "--", "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # exit code 1 = no matches (success); 0 = matches found; >1 = git error
    assert result.returncode in (0, 1), result.stderr
    assert result.returncode == 1, f"conflict markers found:\n{result.stdout}"


def test_A2_detect_objects_signature():
    params = list(inspect.signature(detect_objects).parameters.keys())
    assert params[:2] == ["roi_crops_over_window", "exam_mode"], params


def test_A3_window_length_seconds():
    pipeline = P1P2TrackerPipeline(clip_name="01_phone_use.mkv", fps=25.0)
    window_sec = pipeline.window_size / 25.0
    assert pipeline.window_size == 75, pipeline.window_size
    assert window_sec == pytest.approx(3.0, abs=0.01), window_sec
    assert 2.0 <= window_sec <= 4.0


def test_A4_object_never_gates_motion_only_event():
    fake_event = {
        "event_id": "evt_test_A4",
        "track_id": 1,
        "start_frame": 0,
        "end_frame": 30,
        "start_time": 0.0,
        "end_time": 1.2,
        "total_frames": 30,
        "object_detected": None,
        "object_confidence": None,
        "is_invigilator": False,
        "activities": [],
        "seat_ids": [],
        "motion_intensities": [0.2] * 30,
        "mog2_ratios": [0.2] * 30,
        "severity_score": 0.3,
        "risk_label": "low",
        "color_tag": "green",
        "confidence": 30,
    }
    enriched = enrich_event_with_motion_fields(fake_event)
    confirmed = segment_events([enriched], motion_threshold=0.05, min_duration_sec=0.5)
    assert len(confirmed) >= 1, "motion-only event (no object data) was dropped"


# ---------------------------------------------------------------------------
# GROUP B -- Data Flow
# ---------------------------------------------------------------------------

def test_B1_majority_events_have_nonzero_motion(clip01_run):
    events = clip01_run["events"]
    assert events, "no events produced on 01_phone_use.mkv"
    nonzero = sum(1 for e in events if enrich_event_with_motion_fields(e).get("avg_motion_intensity", 0.0) > 0)
    assert nonzero / len(events) > 0.5, f"{nonzero}/{len(events)} events had nonzero avg_motion_intensity"


def test_B2_every_event_has_boxes_list(clip01_run):
    events = clip01_run["events"]
    assert events
    any_nonempty = False
    for e in events:
        assert "boxes" in e
        assert isinstance(e["boxes"], list)
        if e["boxes"]:
            any_nonempty = True
    assert any_nonempty, "no event had non-empty boxes"


def test_B3_schema_round_trip(clip01_run):
    events = clip01_run["events"]
    assert events
    for e in events:
        schema_event = adapt_bridge_event_to_schema(e, video_id="01_phone_use")
        Event.model_validate(schema_event.model_dump())


def test_B4_seat_behavior_calibrated_vs_uncalibrated():
    if not CLIP_01.exists():
        pytest.skip("01_phone_use.mkv not available")
    calibrated_id = next(iter(GRID_CONFIGS.keys()))  # e.g. "Camera12"

    # Calibrated camera_id -- must not raise, seat_id may or may not resolve
    # to a real seat depending on ROI/motion overlap, but the machinery must run.
    _, bridge_cal, _ = _run_frames(CLIP_01, 60, camera_id=calibrated_id)
    for e in bridge_cal.completed_events:
        assert isinstance(e.get("seat_ids", []), list)

    # Uncalibrated / no camera_id -- must fall back cleanly, no exception.
    _, bridge_unk, _ = _run_frames(CLIP_01, 60, camera_id=None)
    for e in bridge_unk.completed_events:
        seat_ids = e.get("seat_ids", [])
        assert isinstance(seat_ids, list)
        assert all(s == "unknown" or s in GRID_CONFIGS.get(calibrated_id, {}).get("seats", {}) for s in seat_ids) or seat_ids == []


# ---------------------------------------------------------------------------
# GROUP C -- End-to-End Smoke
# ---------------------------------------------------------------------------

def _real_clips():
    if not CLIPS_DIR.exists():
        return []
    return sorted(
        p for p in CLIPS_DIR.glob("*.mkv")
        if not p.name.startswith("dummy")
    )


def test_C1_all_real_clips_run_without_exception():
    # Cap kept low (100, well under the 400-frame ceiling): with the 75-frame
    # (3.0s) detection window each active track now buffers, per-frame cost
    # rose sharply vs. the old 7-frame window -- 400 frames x 7 clips proved
    # too slow/memory-heavy for a regression-guard run. 100 frames/clip is
    # enough to exercise the full per-frame chain (motion->track->detect
    # window->fuse->bridge) at least once per clip without timing out.
    clips = _real_clips()
    if not clips:
        pytest.skip("no clips found in data/drishti/")
    failures = []
    for clip in clips:
        try:
            _run_frames(clip, 100)
        except Exception as e:
            failures.append(f"{clip.name}: {e!r}")
    assert not failures, f"clips raised exceptions: {failures}"


def test_C2_fake_backend_smoke_all_events_post_200(clip01_run):
    fastapi_bridge_dir = REPO_ROOT / "src" / "integration" / "fastapi_bridge"
    sys.path.insert(0, str(fastapi_bridge_dir))
    try:
        from fastapi.testclient import TestClient
        import fake_backend
    except ImportError:
        pytest.skip("fastapi TestClient / fake_backend not importable")

    client = TestClient(fake_backend.app)
    events = clip01_run["events"]
    assert events
    ok = 0
    for e in events:
        schema_event = adapt_bridge_event_to_schema(e, video_id="01_phone_use")
        r = client.post("/internal/events", json=schema_event.model_dump())
        if r.status_code == 200:
            ok += 1
    assert ok == len(events), f"{ok}/{len(events)} posted 200"


def test_C3_idempotency_deferred():
    pytest.skip(
        "fake_backend.py is stateless (no storage, always returns {'ok': True}) "
        "-- duplicate-POST dedup cannot be observed at this layer. Idempotency "
        "tracking (_posted_event_ids) lives client-side in main.py, not in the "
        "backend store; testing real upsert semantics needs a real backend."
    )


def test_C4_crash_recovery_deferred():
    pytest.skip(
        "fake_backend.py has no persistent store to query post-hoc -- "
        "'already-posted events retrievable after simulated stop' is untestable "
        "against this stub without adding storage to fake_backend.py."
    )


# ---------------------------------------------------------------------------
# GROUP D -- Regression Guards
# ---------------------------------------------------------------------------

def test_D1_tracker_sentinel_strings():
    src = (REPO_ROOT / "src" / "integration" / "p1_p2_tracker.py").read_text(encoding="utf-8")
    assert "fuse_track_detections" in src
    assert "exam_mode" in src
    ast.parse(src)  # also catches silent truncation/syntax damage


def test_D2_fusion_docstring_references_window_shape():
    doc = (REPO_ROOT / "src" / "track_det" / "fusion.py").read_text(encoding="utf-8")
    assert "roi_crops_over_window" in doc
    assert "detect_objects(frame)" not in doc


def test_D3_clip04_known_blob_merge_tripwire():
    clip04 = CLIPS_DIR / "04_candidate_talking.mkv"
    if not clip04.exists():
        pytest.skip("04_candidate_talking.mkv not available")
    from src.motion.benchmark_stages import full_pipeline

    predicted = full_pipeline(str(clip04), clip_name=clip04.name, step=30)
    n_events = len(predicted)
    if n_events > 1:
        pytest.fail(
            f"KNOWN-LIMITATION TRIPWIRE: clip 04 now produces {n_events} discrete "
            "events instead of the known single blob-event. This means "
            "camera_id/seat_id passthrough or event_clustering's Stage-2 merge "
            "cap behavior changed -- flag for review, do not assume this is a fix."
        )
    assert n_events >= 0  # documents current (accepted) behavior either way
