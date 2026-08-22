"""
test_object_never_gates.py

Verifies that object detection is a severity booster, NEVER a hard gate.
Constructs an event with valid motion + MOG2 fusion but null/False object detection
(object_detected=False, object_confidence=None), and asserts that a valid Event still emits.
"""

import unittest
from typing import Dict, Any

from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import enrich_event_with_motion_fields, segment_events, cluster_incidents

class TestObjectNeverGates(unittest.TestCase):

    def test_null_object_detection_emits_valid_event(self):
        # 1. Simulate P2P3Bridge track with 0 object detections (object_detected = False)
        bridge = P2P3Bridge(missing_threshold=5, fps=25.0)

        # Process 30 frames of motion track with no object detection hits
        fused_tracks = [
            {
                "track_id": 1,
                "box": (100.0, 100.0, 200.0, 200.0),
                "seat_id": "seat_61",
                "class": None,  # No object class detected
                "confidence": None,  # No object confidence
                "invigilator_flag": False,
            }
        ]

        for frame_idx in range(30):
            bridge.process_fused_tracks(
                fused_tracks=fused_tracks,
                frame_index=frame_idx,
                motion_intensity=0.10,  # Valid motion intensity
            )

        bridge.flush()
        completed = bridge.get_completed_events()

        self.assertEqual(len(completed), 1, "P2P3Bridge failed to produce a completed event for non-object track!")
        raw_event = completed[0]

        # Verify object detection fields in raw event
        self.assertFalse(raw_event["object_detected"], "object_detected should be False")
        self.assertNotIn("metadata", raw_event, "Metadata should be empty/absent when no object detected")

        # 2. Enrich event with motion fields
        motion_stats = {
            "avg_motion_intensity": 0.12,
            "peak_intensity": 0.25,
            "mog2_foreground_ratio": 0.15,
            "intervention_detected": False,
        }
        enriched = enrich_event_with_motion_fields(raw_event, motion_stats)

        # 3. Pass through segmentation (hysteresis filtering)
        confirmed = segment_events([enriched], motion_threshold=0.05, min_duration_sec=0.5)
        self.assertEqual(len(confirmed), 1, "segment_events gated/dropped event because object_detected was False!")

        # 4. Pass through clustering
        clusters = cluster_incidents(confirmed, time_gap_sec=5.0)
        self.assertEqual(len(clusters), 1, "cluster_incidents failed to retain event without object detection!")

        final_event = clusters[0][0]

        # Assert final event is valid and emitted
        self.assertEqual(final_event["event_id"], "event_1")
        self.assertFalse(final_event["object_detected"])
        self.assertGreater(final_event["severity_score"], 0.0, "Severity score should be non-zero from motion signals even with null object detection!")
        self.assertIn("risk_label", final_event)


if __name__ == "__main__":
    unittest.main()
