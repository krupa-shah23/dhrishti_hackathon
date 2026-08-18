"""
test_baseline.py
Owner: P1 - Motion & Calibration

Unit and integration tests for src/motion/baseline.py.
Uses synthetic videos wherever possible; one real-footage integration test
runs against data/drishti/04.CCTV Candidate Talking.mkv if present.
"""

import os
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.motion.baseline import (
    calibrate_baseline,
    compute_calib_window,
    _seat_intensity,
    _robust_stats,
)
from src.motion.grid_config import GRID_CONFIGS

# ── helpers ────────────────────────────────────────────────────────────────

def _make_synthetic_video(path: str, fps: float, n_frames: int,
                           width: int = 640, height: int = 480,
                           fill_value: int = 80) -> None:
    """Write a short synthetic video filled with a uniform grey level."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), fill_value, dtype=np.uint8)
        out.write(frame)
    out.release()


def _make_motion_video(path: str, fps: float, n_frames: int,
                        width: int = 640, height: int = 480) -> None:
    """
    Write a video where frames alternate between background (dark) and a
    bright patch in the bottom-right region to simulate motion in seat_60.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Brief bright flash as rare outliers (only twice in 80 frames)
        if i in (10, 40):
            frame[340:480, 280:480] = 200
        out.write(frame)
    out.release()


REAL_VIDEO = Path("data/drishti/04.CCTV Candidate Talking.mkv")
CAMERA_ID = "Camera12"


# ── test cases ──────────────────────────────────────────────────────────────

class TestComputeCalibWindow(unittest.TestCase):

    def test_short_video_gives_60s(self):
        # 143.12 s  →  10% = 14.31 < 60  →  clamped to 60
        self.assertEqual(compute_calib_window(143.12), 60.0)

    def test_very_long_video_gives_300s(self):
        # 10% of 4000 s = 400 > 300  →  clamped to 300
        self.assertEqual(compute_calib_window(4000.0), 300.0)

    def test_medium_video_proportional(self):
        # 10% of 1000 s = 100  →  between 60 and 300
        self.assertEqual(compute_calib_window(1000.0), 100.0)

    def test_tiny_video_gives_60s(self):
        self.assertEqual(compute_calib_window(10.0), 60.0)


class TestSeatIntensity(unittest.TestCase):

    def test_empty_mask_gives_zero(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        self.assertEqual(_seat_intensity(mask, 0, 0, 100, 100), 0.0)

    def test_full_mask_gives_one(self):
        mask = np.full((480, 640), 255, dtype=np.uint8)
        self.assertAlmostEqual(_seat_intensity(mask, 0, 0, 640, 480), 1.0)

    def test_half_mask_gives_half(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, :50] = 255  # left half lit
        result = _seat_intensity(mask, 0, 0, 100, 100)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_zero_area_seat_is_safe(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # x1 == x2 → zero area
        result = _seat_intensity(mask, 50, 50, 50, 80)
        self.assertEqual(result, 0.0)


class TestRobustStats(unittest.TestCase):

    def test_empty_returns_zero(self):
        mean, std = _robust_stats([])
        self.assertEqual(mean, 0.0)
        self.assertEqual(std, 0.0)

    def test_single_value(self):
        mean, std = _robust_stats([0.5])
        self.assertAlmostEqual(mean, 0.5)
        self.assertEqual(std, 0.0)

    def test_outliers_resisted(self):
        # 9 low values + 1 very high outlier
        values = [0.01] * 9 + [0.99]
        mean_robust, _ = _robust_stats(values)
        mean_raw = float(np.mean(values))
        # Robust mean (median) should be much lower than raw mean
        self.assertLess(mean_robust, mean_raw)
        self.assertAlmostEqual(mean_robust, 0.01)

    def test_uniform_distribution(self):
        values = [0.1] * 100
        mean, std = _robust_stats(values)
        self.assertAlmostEqual(mean, 0.1, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)


class TestCalibrateBaselineSynthetic(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.tmpdir.name, "test.mp4")
        # 8 FPS, 80 frames  ≈  10 seconds — well within 60 s calib window
        _make_synthetic_video(self.video_path, fps=8.0, n_frames=80)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_every_seat_receives_stats(self):
        result = calibrate_baseline(self.video_path, CAMERA_ID,
                                    calib_window_sec=60.0)
        expected_seats = set(GRID_CONFIGS[CAMERA_ID]["seats"].keys())
        self.assertEqual(set(result.keys()), expected_seats)

    def test_no_global_baseline(self):
        result = calibrate_baseline(self.video_path, CAMERA_ID,
                                    calib_window_sec=60.0)
        for seat_id, stats in result.items():
            self.assertIsInstance(stats, tuple)
            self.assertEqual(len(stats), 2)
            self.assertTrue(seat_id.startswith("seat_"),
                            f"Unexpected seat key: {seat_id}")

    def test_results_are_finite(self):
        result = calibrate_baseline(self.video_path, CAMERA_ID,
                                    calib_window_sec=60.0)
        for seat_id, (mean, std) in result.items():
            self.assertTrue(math.isfinite(mean),
                            f"{seat_id} mean is not finite")
            self.assertTrue(math.isfinite(std),
                            f"{seat_id} std is not finite")

    def test_results_are_deterministic(self):
        r1 = calibrate_baseline(self.video_path, CAMERA_ID,
                                 calib_window_sec=60.0, sample_rate=1.0)
        r2 = calibrate_baseline(self.video_path, CAMERA_ID,
                                 calib_window_sec=60.0, sample_rate=1.0)
        for seat_id in r1:
            self.assertAlmostEqual(r1[seat_id][0], r2[seat_id][0], places=10)
            self.assertAlmostEqual(r1[seat_id][1], r2[seat_id][1], places=10)

    def test_outlier_movement_does_not_dominate(self):
        """
        Create a video with occasional large motion bursts.
        Robust mean should be lower than raw mean for the affected seat.
        """
        motion_path = os.path.join(self.tmpdir.name, "motion.mp4")
        _make_motion_video(motion_path, fps=8.0, n_frames=80)

        # compute robust
        result_robust = calibrate_baseline(motion_path, CAMERA_ID,
                                             calib_window_sec=60.0,
                                             sample_rate=1.0)
        
        # for seat_60 (the flash region) robust mean should be approx 0 (since mostly dark)
        self.assertLessEqual(result_robust["seat_60"][0], 0.05)


    def test_empty_video_path_returns_empty(self):
        result = calibrate_baseline("nonexistent_video.mp4", CAMERA_ID)
        self.assertEqual(result, {})

    def test_unknown_camera_returns_empty(self):
        result = calibrate_baseline(self.video_path, "UnknownCamera")
        self.assertEqual(result, {})


class TestCalibrateBaselineRealVideo(unittest.TestCase):
    """Integration test against the actual DRISHTI Camera12 footage."""

    @unittest.skipUnless(REAL_VIDEO.exists(), "Real DRISHTI video not found.")
    def test_real_calibration(self):
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(REAL_VIDEO))
        source_fps = cap.get(_cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertAlmostEqual(source_fps, 8.0, places=1)
        duration = frame_count / source_fps
        self.assertAlmostEqual(duration, 143.12, delta=1.0)

        result = calibrate_baseline(str(REAL_VIDEO), CAMERA_ID)

        expected_seats = set(GRID_CONFIGS[CAMERA_ID]["seats"].keys())
        self.assertEqual(set(result.keys()), expected_seats)

        for seat_id, (mean, std) in result.items():
            with self.subTest(seat=seat_id):
                self.assertTrue(math.isfinite(mean))
                self.assertTrue(math.isfinite(std))
                self.assertGreaterEqual(mean, 0.0)
                self.assertGreaterEqual(std, 0.0)


if __name__ == "__main__":
    unittest.main()
