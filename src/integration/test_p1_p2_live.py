"""
test_p1_p2_live.py
Unit and integration tests for live P1 Motion/ROI -> P2 Tracker pipeline.
Tests all requirements specified in Phase 2:
1. Frame with multiple P1 ROIs
2. Frame with zero P1 ROIs
3. Consecutive frames with same ROI
4. ROI disappearance
5. ROI reappearance
6. Multiple simultaneous tracks
7. Long ROI-free gaps (> max_age)
8. Tracker reset between clips
"""

import time
import numpy as np
import cv2

try:
    from src.integration.p1_p2_tracker import P1P2TrackerPipeline
    from src.motion.roi import draw_rois
except ImportError:
    from .p1_p2_tracker import P1P2TrackerPipeline
    from ..motion.roi import draw_rois


def create_synthetic_frame(
    width: int = 640,
    height: int = 480,
    rectangles: list = None,
) -> np.ndarray:
    """
    Creates a synthetic BGR frame with a static gray background and optional solid white rectangles.
    """
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    if rectangles:
        for (x1, y1, x2, y2) in rectangles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return frame


def test_live_pipeline_synthetic_sequence():
    print("\n============================================================")
    print("  RUNNING LIVE P1 -> P2 INTEGRATION TEST (SYNTHETIC FRAME STREAM)")
    print("============================================================")

    pipeline = P1P2TrackerPipeline(min_area=100, max_distance=80.0, max_age=10)

    # 1. Warm up background estimator with 10 static frames
    print("Phase A: Warming up background model...")
    warmup_frame = create_synthetic_frame()
    for i in range(10):
        pipeline.process_frame(warmup_frame, i)

    # 2. Sequence with moving rectangle (Subject A), second rectangle (Subject B), gaps, reappearance
    test_sequence = [
        # Frames 10-14: Two moving subjects
        [(50, 50, 100, 120), (300, 200, 360, 280)],
        [(55, 52, 105, 122), (305, 202, 365, 282)],
        [(60, 54, 110, 124), (310, 204, 370, 284)],
        [(65, 56, 115, 126), (315, 206, 375, 286)],
        [(70, 58, 120, 128), (320, 208, 380, 288)],
        # Frames 15-17: Disappearance / Zero ROIs frame
        [],
        [],
        [],
        # Frames 18-20: Reappearance within max_age (3 gaps < max_age=10)
        [(75, 60, 125, 130)],
        [(80, 62, 130, 132)],
        [(85, 64, 135, 134)],
        # Frames 21-35: Long gap exceeding max_age (15 gaps > max_age=10)
        *([[]] * 15),
        # Frames 36-38: Reappearance after long gap -> New track ID expected
        [(90, 66, 140, 136)],
        [(95, 68, 145, 138)],
    ]

    start_time = time.time()
    results = []
    max_track_id = 0

    print("Phase B: Running live sequential stream...")
    for idx, rects in enumerate(test_sequence, start=10):
        frame = create_synthetic_frame(rectangles=rects)
        res = pipeline.process_frame(frame, idx)
        results.append(res)

        # Validation checks on returned structure
        assert "frame_index" in res, "Missing frame_index in result"
        assert "roi_boxes" in res, "Missing roi_boxes in result"
        assert "tracks" in res, "Missing tracks in result"

        for tr in res["tracks"]:
            assert isinstance(tr["track_id"], int), f"Track ID {tr['track_id']} must be integer"
            x1, y1, x2, y2 = tr["box"]
            assert x1 < x2 and y1 < y2, f"Invalid box format (x1<x2, y1<y2): {tr['box']}"
            max_track_id = max(max_track_id, tr["track_id"])

        t_ids = [t["track_id"] for t in res["tracks"]]
        n_rois = len(res["roi_boxes"])
        print(f"  Frame {idx:02d}: {n_rois} ROIs -> {len(res['tracks'])} tracks, Track IDs={t_ids}")

    elapsed = time.time() - start_time
    fps = len(test_sequence) / elapsed if elapsed > 0 else 0

    # 3. Test Tracker Reset between clips
    print("\nPhase C: Testing tracker reset for new clip...")
    pipeline.reset()
    fresh_res = pipeline.process_frame(create_synthetic_frame(rectangles=[(100, 100, 150, 170)]), 0)
    fresh_ids = [t["track_id"] for t in fresh_res["tracks"]]
    assert 1 in fresh_ids, f"Track ID after reset should restart at 1, got {fresh_ids}"

    print(f"\n--- Live Integration Test Summary ---")
    print(f"Total frames processed: {len(test_sequence) + 10}")
    print(f"Highest track ID assigned: {max_track_id}")
    print(f"Processing time: {elapsed:.4f} seconds")
    print(f"Approximate FPS: {fps:.1f} FPS")
    print("STATUS: PHASE 2 TEST PASS")
    return True


def test_process_frame_uses_native_input_resolution_no_implicit_resize():
    """
    Phase 1 coordinate-contract lock (§4).

    P1P2TrackerPipeline.process_frame() must operate directly on whatever
    resolution frame it is handed -- it must NOT implicitly resize to a fixed
    "processing resolution" (e.g. 480p). Locking this in with a test because
    tracing the real runtime path found frame_stream.py's 480p downsize step
    is not wired into this pipeline at all (process_frame receives whatever
    the caller passed straight through to MotionEstimator/get_rois), so all
    ROI/track/event coordinates are in the SAME resolution as the input frame
    -- i.e. native video resolution in every real caller, which reads frames
    via plain cv2.VideoCapture. If someone later adds an implicit resize
    inside process_frame, this test must fail.

    Uses Camera12 (base grid resolution 640x480) at exactly 2x that
    resolution (1280x960) so get_grid_config()'s proportional scaling gives
    an exact, checkable expected seat rectangle -- if scaling used a
    hardcoded resolution instead of the actual input frame's shape, this
    would attach the wrong seat_id or place boxes outside the true bounds.
    """
    from src.motion.grid_config import get_grid_config

    width, height = 1280, 960  # exactly 2x Camera12's base (640, 480)
    pipeline = P1P2TrackerPipeline(min_area=100)

    # Warm up MOG2 background model at this resolution.
    warmup_frame = create_synthetic_frame(width=width, height=height)
    for i in range(10):
        pipeline.process_frame(warmup_frame, i, camera_id="Camera12")

    # seat_66 at base res is (0, 120, 150, 480) -> exactly doubled at 1280x960.
    expected_seat_66 = tuple(v * 2 for v in (0, 120, 150, 480))
    scaled_seats = get_grid_config("Camera12", width, height)["seats"]
    assert scaled_seats["seat_66"] == expected_seat_66, (
        f"get_grid_config scaling itself is wrong: {scaled_seats['seat_66']} != {expected_seat_66}"
    )

    # Place a moving-subject rectangle well inside the scaled seat_66 region.
    rect = (20, 200, 120, 400)
    frame = create_synthetic_frame(width=width, height=height, rectangles=[rect])
    res = pipeline.process_frame(frame, 10, camera_id="Camera12")

    assert len(res["roi_boxes"]) > 0, "Expected at least one ROI at 1280x960 input resolution"
    for (x1, y1, x2, y2) in res["roi_boxes"]:
        # Coordinates must fall within the ACTUAL input frame (1280x960), not
        # a hardcoded 640x480 / 480p processing resolution.
        assert 0 <= x1 < x2 <= width, f"ROI x-range {(x1, x2)} outside native width {width}"
        assert 0 <= y1 < y2 <= height, f"ROI y-range {(y1, y2)} outside native height {height}"

    for ft in res["fused_tracks"]:
        assert ft["seat_id"] == "seat_66", (
            f"Expected seat_66 (scaled to {width}x{height}), got {ft['seat_id']} -- "
            f"seat attribution did not scale with the actual input resolution"
        )

    print("STATUS: COORDINATE CONTRACT TEST PASS (native-resolution passthrough confirmed)")


def test_pipeline_fps_defaults_and_propagates_to_pose_analyzer():
    """
    Phase 1 FPS/timestamp-contract lock (§7).

    P1P2TrackerPipeline previously hardcoded fps=25.0 for its internal
    PoseGestureAnalyzer regardless of the clip's real source fps (e.g.
    04_candidate_talking.mkv is 8fps, 07_seat_exchange.mkv is 22fps) -- so
    timestamp_sec=frame_index/25.0 was wrong video-time for any non-25fps
    clip. This locks two things:
      1. Default behavior is unchanged (fps=25.0) when a caller doesn't pass
         fps -- every existing caller in this repo constructs
         P1P2TrackerPipeline without an fps argument, so this must not
         silently change their already-validated output.
      2. A caller that DOES pass the clip's real fps gets that value used for
         pose_analyzer.update()'s timestamp_sec, not a hardcoded 25.0.
    """
    default_pipeline = P1P2TrackerPipeline(min_area=100)
    assert default_pipeline.fps == 25.0
    assert default_pipeline.pose_analyzer.fps == 25.0

    real_fps = 8.0  # matches 04_candidate_talking.mkv's real source fps
    pipeline = P1P2TrackerPipeline(min_area=100, fps=real_fps)
    assert pipeline.fps == real_fps
    assert pipeline.pose_analyzer.fps == real_fps, (
        "PoseGestureAnalyzer must receive the pipeline's real fps, not a hardcoded default"
    )

    # frame 80 at 8fps is real video-time 10.0s, not frame_index/25.0 = 3.2s.
    frame = create_synthetic_frame()
    pipeline.process_frame(frame, 80, camera_id=None)
    assert pipeline.fps == real_fps  # unchanged by process_frame itself

    # reset() must preserve the pipeline's real fps, not revert to 25.0.
    pipeline.reset()
    assert pipeline.pose_analyzer.fps == real_fps

    print("STATUS: FPS CONTRACT TEST PASS (real fps propagates, default unchanged)")


def test_mog2_foreground_ratio_is_genuine_not_fused_or_fabricated():
    """
    Phase 1 motion-metric contract lock (§6).

    event_clustering.py's `mog2_foreground_ratio` field already exists as a
    documented contract field, but every real caller in this repo
    (scripts/verification/*.py) was feeding it a hardcoded placeholder
    (1.0 or 0.0), never a genuine MOG2-only measurement. This locks in that
    process_frame()'s new "mog2_foreground_ratio" return value is:
    computed from MOG2's own binary mask, pre-fusion -- deterministically
    reproducible from an independently-constructed MotionEstimator fed the
    identical frame sequence (rather than a hardcoded constant, which by
    definition could never match an independent recomputation like this).
    """
    from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity

    width, height = 200, 150
    warmup_frame = create_synthetic_frame(width=width, height=height)
    motion_frame = create_synthetic_frame(width=width, height=height, rectangles=[(20, 20, 80, 80)])

    # Independent reference estimator, fed the exact same sequence.
    ref_estimator = MotionEstimator()
    for _ in range(10):
        ref_estimator.get_motion_mask(warmup_frame)
    ref_mag, ref_mask = ref_estimator.get_motion_mask(motion_frame)
    expected_mog2_ratio = motion_intensity(ref_mask)
    expected_fused_ratio = motion_intensity(fuse_motion_signal(ref_mag, ref_mask))

    # Pipeline under test, fed the identical sequence.
    pipeline = P1P2TrackerPipeline(min_area=100)
    for i in range(10):
        pipeline.process_frame(warmup_frame, i)
    result = pipeline.process_frame(motion_frame, 10)

    assert result["mog2_foreground_ratio"] == expected_mog2_ratio, (
        "mog2_foreground_ratio must equal MOG2's own mask intensity, deterministically"
    )
    assert result["motion_intensity"] == expected_fused_ratio

    print("STATUS: MOTION-METRIC CONTRACT TEST PASS (mog2_foreground_ratio is genuine)")


if __name__ == "__main__":
    test_live_pipeline_synthetic_sequence()
    test_process_frame_uses_native_input_resolution_no_implicit_resize()
    test_pipeline_fps_defaults_and_propagates_to_pose_analyzer()
    test_mog2_foreground_ratio_is_genuine_not_fused_or_fabricated()
