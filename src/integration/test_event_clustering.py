"""
§1 + §2 combined: 
  - Adjacency test (clip-4 synthetic proxy, per Day 4 plan)
  - Joint smoke test on 07_seat_exchange.mkv event windows
"""
import sys
import unittest
sys.path.insert(0, '.')

from src.integration.event_clustering import (
    enrich_event_with_motion_fields,
    segment_events,
    cluster_incidents,
    link_related_events,
)


class TestClip4AdjacencyProxy(unittest.TestCase):
    """
    Synthetic proxy for clip-4 (two talking bursts from same candidate, ~60s apart).
    Clip-4 filename mapping unconfirmed as of Day 4 — running synthetic proxy per
    §1 specification. Label: SYNTHETIC PROXY TEST.
    
    Ground truth: two motion events from the same spatial source (seat region ~(200,300)),
    ~60s apart. Must link via related_event_id, must NOT be merged into one cluster
    (they are separated by more than the cluster time_gap_sec).
    """

    def _make_event(self, event_id, start_time, end_time, cx, cy):
        """Build a minimal enriched event mimicking P2P3Bridge output."""
        return {
            "event_id": event_id,
            "track_id": int(event_id.split("_")[1]),
            "positions": [(cx, cy)] * 5,  # same spatial centroid repeated
            "boxes": [(cx - 50, cy - 50, 100, 100)] * 5,
            "frame_indices": list(range(5)),
            "start_frame": int(start_time * 25),
            "end_frame": int(end_time * 25),
            "start_time": start_time,
            "end_time": end_time,
            "total_frames": int((end_time - start_time) * 25),
            "object_detected": False,
            "is_invigilator": False,
            "metadata": [],
            # Motion stats provided directly (would come from pipeline in production)
            "avg_motion_intensity": 0.12,
            "peak_intensity": 0.25,
            "mog2_foreground_ratio": 0.90,
            "invigilator_excluded": False,
            "intervention_detected": False,
            "event_type": "talking",
            "segment_confirmed": True,
        }

    def test_related_event_id_links_two_bursts(self):
        """
        Two talking bursts from same seat (cx~200, cy~300), 60s apart.
        Expected: both get related_event_id pointing to each other.
        Must NOT be merged into one cluster (gap > 5s).
        """
        ev_a = self._make_event("event_1", start_time=45.0, end_time=55.0, cx=200, cy=300)
        ev_b = self._make_event("event_2", start_time=108.0, end_time=118.0, cx=210, cy=295)

        # 1. Cluster — must produce 2 separate clusters (60s gap >> 5s threshold)
        clusters = cluster_incidents([ev_a, ev_b], time_gap_sec=5.0)
        self.assertEqual(len(clusters), 2,
                         f"Expected 2 separate clusters (60s gap), got {len(clusters)}")
        print(f"  cluster_incidents: {len(clusters)} clusters — PASS")

        # 2. Link — must link across clusters
        linked = link_related_events(
            clusters, spatial_radius_px=150.0, time_window_sec=120.0
        )
        ev_a_out = next(e for e in linked if e["event_id"] == "event_1")
        ev_b_out = next(e for e in linked if e["event_id"] == "event_2")

        self.assertEqual(ev_a_out.get("related_event_id"), "event_2",
                         "event_1 should have related_event_id=event_2")
        self.assertEqual(ev_b_out.get("related_event_id"), "event_1",
                         "event_2 should have related_event_id=event_1")
        print(f"  event_1.related_event_id = {ev_a_out.get('related_event_id')} — PASS")
        print(f"  event_2.related_event_id = {ev_b_out.get('related_event_id')} — PASS")

    def test_no_link_for_spatially_distant_events(self):
        """
        Two events far apart spatially (>150px) must NOT be linked,
        even if temporally close.
        """
        ev_a = self._make_event("event_3", start_time=45.0, end_time=55.0, cx=50, cy=50)
        ev_b = self._make_event("event_4", start_time=110.0, end_time=120.0, cx=800, cy=600)

        clusters = cluster_incidents([ev_a, ev_b], time_gap_sec=5.0)
        linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=120.0)

        ev_a_out = next(e for e in linked if e["event_id"] == "event_3")
        self.assertIsNone(ev_a_out.get("related_event_id"),
                          "Spatially distant events must not be linked")
        print(f"  Distant events: related_event_id={ev_a_out.get('related_event_id')} — PASS (None)")

    def test_events_in_same_cluster_not_cross_linked(self):
        """
        Two events in the SAME cluster (< 5s gap) must not get related_event_id
        from link_related_events (they're already merged).
        """
        ev_a = self._make_event("event_5", start_time=45.0, end_time=55.0, cx=200, cy=300)
        ev_b = self._make_event("event_6", start_time=58.0, end_time=68.0, cx=205, cy=298)

        clusters = cluster_incidents([ev_a, ev_b], time_gap_sec=5.0)
        self.assertEqual(len(clusters), 1, "Events 3s apart should be one cluster")

        linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=120.0)
        ev_a_out = next(e for e in linked if e["event_id"] == "event_5")
        self.assertIsNone(ev_a_out.get("related_event_id"),
                          "Events in same cluster must not be cross-linked")
        print(f"  Same cluster: related_event_id={ev_a_out.get('related_event_id')} — PASS (None)")


class TestSignalFormatCompatibility(unittest.TestCase):
    """Verify our enriched event dict matches P3's expected field schema exactly."""

    def test_all_required_fields_present(self):
        raw_event = {
            "event_id": "event_10",
            "track_id": 10,
            "positions": [(100, 200)],
            "boxes": [(75, 175, 50, 50)],
            "frame_indices": [0],
            "start_frame": 0,
            "end_frame": 25,
            "start_time": 0.0,
            "end_time": 1.0,
            "total_frames": 25,
            "object_detected": True,
            "is_invigilator": False,
            "metadata": [{"frame": 5, "class": "phone", "confidence": 0.87}],
        }
        motion_stats = {
            "avg_motion_intensity": 0.18,
            "peak_intensity": 0.41,
            "mog2_foreground_ratio": 0.95,
            "intervention_detected": False,
        }

        enriched = enrich_event_with_motion_fields(raw_event, motion_stats)

        # Verify all P3-required fields are present
        required_fields = [
            "avg_motion_intensity", "peak_intensity", "mog2_foreground_ratio",
            "invigilator_excluded", "intervention_detected", "event_type"
        ]
        for field in required_fields:
            self.assertIn(field, enriched, f"Missing required field: {field}")

        self.assertAlmostEqual(enriched["avg_motion_intensity"], 0.18)
        self.assertAlmostEqual(enriched["peak_intensity"], 0.41)
        self.assertAlmostEqual(enriched["mog2_foreground_ratio"], 0.95)
        self.assertFalse(enriched["invigilator_excluded"])
        self.assertFalse(enriched["intervention_detected"])
        self.assertEqual(enriched["event_type"], "phone_use")  # phone detected in metadata
        print(f"  All P3 fields present. event_type={enriched['event_type']} — PASS")


    def _make_event(self, event_id, start_time, end_time, cx, cy):
        return {
            "event_id": event_id,
            "start_time": start_time,
            "end_time": end_time,
            "avg_motion_intensity": 0.1,
            "positions": [{"centroid_x": cx, "centroid_y": cy}],
            "object_detected": True,
            "is_invigilator": False
        }

    def test_activities_field_preservation(self):
        ev_1 = self._make_event("event_1", start_time=10.0, end_time=20.0, cx=200, cy=300)
        ev_1['activities'] = ["test"]
        
        # 1. segmentation
        segmented = segment_events([ev_1])
        self.assertEqual(len(segmented), 1)
        self.assertEqual(segmented[0]['activities'], ["test"])
        
        # 2. clustering
        clusters = cluster_incidents(segmented)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0][0]['activities'], ["test"])
        
        # 3. linking
        linked = link_related_events([clusters[0]])
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]['activities'], ["test"])

if __name__ == "__main__":
    import unittest
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
