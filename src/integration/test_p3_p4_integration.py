"""
test_p3_p4_integration.py
Integration tests for Phase 3 (Object Detection + Fusion) and Phase 4 (Invigilator Filtering).

Tests all prompt requirements:
- TEST A: 1 frame with 2 ROIs and 2 detections -> 2 tracks produced and fused without error
- TEST B: 1 frame with 0 ROIs but detections present -> tracker receives [] and does not crash
- TEST C: 1 frame with ROIs but no detections -> tracks remain valid with class=None, confidence=None
- TEST D: 1 frame with 0 ROIs and 0 detections -> tracker receives [] and fusion returns safely
- TEST E: Several sequential frames -> track IDs remain stable for moving subjects
- TEST F: Detection matches track -> class/confidence attached according to fusion
- TEST G: Unmatched detection -> does not create a fake track
- TEST H: Invigilator filter -> insufficient history is False; patrolling history is True
- TEST I: Robustness against detector exceptions

Run from repo root:
    python -m src.integration.test_p3_p4_integration
"""

import unittest
from unittest.mock import patch
import numpy as np

try:
    from src.integration.p1_p2_tracker import P1P2TrackerPipeline
    from src.track_det.tracker import reset_tracker
except ImportError:
    from .p1_p2_tracker import P1P2TrackerPipeline
    from ..track_det.tracker import reset_tracker


def make_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


class TestPhase3Phase4Integration(unittest.TestCase):
    def setUp(self):
        reset_tracker()
        self.pipeline = P1P2TrackerPipeline(min_area=100, max_distance=80.0, max_age=10)
        self.dummy_frame = make_dummy_frame()

    def test_A_two_rois_two_detections(self):
        """TEST A: 1 frame with 2 ROIs and 2 detections."""
        rois = [(50.0, 50.0, 150.0, 200.0), (300.0, 300.0, 400.0, 450.0)]
        dets = [
            ((20.0, 20.0, 50.0, 50.0), "phone", 0.92),
            ((20.0, 20.0, 50.0, 50.0), "paper-chit", 0.88),
        ]
        
        # We call detect_objects per track, so return one detection per call
        mock_returns = [[dets[0]], [dets[1]]]
        def mock_detect(*args, **kwargs):
            return mock_returns.pop(0) if mock_returns else []

        with patch("src.integration.p1_p2_tracker.detect_objects", side_effect=mock_detect):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(len(res["rois"]), 2)
        self.assertEqual(len(res["tracks"]), 2)
        self.assertEqual(len(res["detections"]), 2)
        self.assertEqual(len(res["fused_tracks"]), 2)

        # Check matched detections in fused tracks
        classes = sorted([ft["class"] for ft in res["fused_tracks"]])
        self.assertEqual(classes, ["paper-chit", "phone"])

    def test_B_zero_rois_with_detections(self):
        """TEST B: 1 frame with 0 ROIs but detections present."""
        rois = []
        dets = [((100.0, 100.0, 130.0, 130.0), "phone", 0.95)]

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=dets):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(res["rois"], [])
        self.assertEqual(res["tracks"], [])
        self.assertEqual(res["detections"], [])
        self.assertEqual(res["fused_tracks"], [])

    def test_C_rois_with_no_detections(self):
        """TEST C: 1 frame with ROIs but no detections."""
        rois = [(50.0, 50.0, 150.0, 200.0)]
        dets = []

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=dets):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(len(res["tracks"]), 1)
        ft = res["fused_tracks"][0]
        self.assertIsNone(ft["class"])
        self.assertIsNone(ft["confidence"])
        self.assertFalse(ft["invigilator_flag"])

    def test_D_zero_rois_zero_detections(self):
        """TEST D: 1 frame with no ROIs and no detections."""
        rois = []
        dets = []

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=dets):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(res["rois"], [])
        self.assertEqual(res["tracks"], [])
        self.assertEqual(res["detections"], [])
        self.assertEqual(res["fused_tracks"], [])

    def test_E_sequential_frames_track_id_stability(self):
        """TEST E: Several sequential frames maintain stable track IDs."""
        sequence = [
            [(50.0, 50.0, 100.0, 100.0)],
            [(55.0, 52.0, 105.0, 102.0)],
            [(60.0, 54.0, 110.0, 104.0)],
            [(65.0, 56.0, 115.0, 106.0)],
        ]

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=[]):
            track_ids = []
            for i, rois in enumerate(sequence):
                res = self.pipeline.process_frame(
                    self.dummy_frame, frame_index=i, rois_override=rois
                )
                self.assertEqual(len(res["tracks"]), 1)
                track_ids.append(res["tracks"][0]["track_id"])

        self.assertEqual(len(set(track_ids)), 1, f"Track ID changed across frames: {track_ids}")

    def test_F_detection_matches_track(self):
        """TEST F: Detection inside track box attaches class & confidence."""
        rois = [(100.0, 100.0, 300.0, 300.0)]
        dets = [((50.0, 50.0, 80.0, 80.0), "phone", 0.91)]

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=dets):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        ft = res["fused_tracks"][0]
        self.assertEqual(ft["class"], "phone")
        self.assertAlmostEqual(ft["confidence"], 0.91)

    def test_G_unmatched_detection_does_not_create_fake_track(self):
        """TEST G: Detection outside any track box does not create a fake track."""
        rois = [(50.0, 50.0, 100.0, 100.0)]
        dets = [((350.0, 350.0, 400.0, 400.0), "phone", 0.99)]

        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=dets):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(len(res["fused_tracks"]), 1)
        ft = res["fused_tracks"][0]
        self.assertIsNone(ft["class"])
        self.assertIsNone(ft["confidence"])

    def test_H_invigilator_filter_integration(self):
        """TEST H: Invigilator filtering over track history."""
        # Phase 1: 5 frames of motion (insufficient history < min_hits=15)
        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=[]):
            for i in range(5):
                rois = [(10.0 + i * 20.0, 10.0, 50.0 + i * 20.0, 50.0)]
                res = self.pipeline.process_frame(
                    self.dummy_frame, frame_index=i, rois_override=rois
                )

        ft = res["fused_tracks"][0]
        self.assertFalse(ft["invigilator_flag"], "Should be False for history length 5 (< 15)")

        # Phase 2: Continue walking up to 20 frames (history length >= 15, high net displacement & path)
        with patch("src.integration.p1_p2_tracker.detect_objects", return_value=[]):
            for i in range(5, 20):
                rois = [(10.0 + i * 20.0, 10.0, 50.0 + i * 20.0, 50.0)]
                res = self.pipeline.process_frame(
                    self.dummy_frame, frame_index=i, rois_override=rois
                )

        ft = res["fused_tracks"][0]
        self.assertTrue(ft["invigilator_flag"], "Should be True for 20-frame patrolling track")

    def test_I_detector_exception_handling(self):
        """TEST I: Exception during detection is caught cleanly and does not break tracking."""
        rois = [(50.0, 50.0, 100.0, 100.0)]

        def failing_detector(crop_list, exam_mode):
            raise RuntimeError("CUDA out of memory or inference hardware failure")

        with patch("src.integration.p1_p2_tracker.detect_objects", side_effect=failing_detector):
            res = self.pipeline.process_frame(
                self.dummy_frame, frame_index=0, rois_override=rois
            )

        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["detections"], [])
        self.assertEqual(len(res["fused_tracks"]), 1)
        self.assertIsNone(res["fused_tracks"][0]["class"])


if __name__ == "__main__":
    unittest.main()
