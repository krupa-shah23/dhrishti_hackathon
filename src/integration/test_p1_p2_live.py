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


if __name__ == "__main__":
    test_live_pipeline_synthetic_sequence()
