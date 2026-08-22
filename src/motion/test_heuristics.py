import unittest
from src.motion.heuristics import (
    is_invigilator_motion, tag_interventions, is_seat_vacated,
    flag_seat_vacant_near_invigilator, build_explanation,
)

class TestInvigilatorMotion(unittest.TestCase):
    def test_genuine_traveling_pattern(self):
        # Travels from seat_1 -> seat_2 -> seat_3
        history = [{'seat_1'}, {'seat_2'}, {'seat_3'}]
        self.assertTrue(is_invigilator_motion(history))

    def test_single_stationary_cell_spike(self):
        # Spikes in one cell for a few frames
        history = [{'seat_1'}, {'seat_1'}, {'seat_1'}]
        self.assertFalse(is_invigilator_motion(history))

    def test_simultaneous_multicell_no_travel(self):
        # Simultaneous motion across all 3 cells, not traveling
        history = [{'seat_1', 'seat_2', 'seat_3'}, {'seat_1', 'seat_2', 'seat_3'}]
        self.assertFalse(is_invigilator_motion(history))

class TestInterventionTagging(unittest.TestCase):
    def test_synthetic_55s_intervention(self):
        # SYNTHETIC PROXY: mimicking clip 1's 55s intervention pattern, not validated real-footage ground truth
        # Event is flagged around 51-55s. Invigilator walks past multiple cells around that time.
        events = [{'start_sec': 51.0, 'end_sec': 55.0, 'intervention_detected': False}]
        
        full_cell_history = [
            (50.0, {'seat_60'}), # Before window
            (52.0, {'seat_61'}), # Start of traveling inside event window
            (53.0, {'seat_63'}),
            (54.0, {'seat_64'}),
            (56.0, {'seat_65'}), # Slightly after, caught by padding
        ]
        
        tag_interventions(events, full_cell_history)
        self.assertTrue(events[0]['intervention_detected'])
        
        # Test non-overlapping event
        events2 = [{'start_sec': 10.0, 'end_sec': 15.0, 'intervention_detected': False}]
        tag_interventions(events2, full_cell_history)
        self.assertFalse(events2[0]['intervention_detected'])

class TestSeatVacated(unittest.TestCase):
    def test_sustained_zero_signal_past_timeout(self):
        # Timeout is 3. 4 consecutive False values.
        history = [True, False, False, False, False]
        self.assertTrue(is_seat_vacated(history, 3))

    def test_brief_pause_then_resumed_activity(self):
        # Timeout is 3. Only 2 consecutive False, then True.
        # It's looking at the end of the history. If history ends in True, it's not vacated.
        history = [True, False, False, True, True]
        self.assertFalse(is_seat_vacated(history, 3))

    def test_edge_case_exactly_at_boundary(self):
        # Timeout is 3. Exactly 3 consecutive False at the end.
        history = [True, False, False, False]
        # Should be False because requirement says "strictly greater than timeout" (exclusive boundary)
        self.assertFalse(is_seat_vacated(history, 3), "Boundary case exactly at timeout should be False")

class TestFlagSeatVacantNearInvigilator(unittest.TestCase):
    """
    §3.4 seat_vacant_near_invigilator: uses Camera12's real adjacency map
    (seat_65 <-> seat_64, see grid_config.py) rather than synthetic seat IDs,
    so Gate 1's adjacency check exercises real, already-verified data.
    """
    CAMERA_ID = "Camera12"
    SEAT_ID = "seat_65"  # adjacent seat per grid_config.py: seat_64

    # A genuine traveling pattern (3+ distinct cells, matches
    # test_genuine_traveling_pattern) that touches seat_64 -- adjacent to
    # seat_65 -- but never seat_65 itself, to prove Gate 1's adjacency path
    # (not just same-seat) is exercised.
    INVIGILATOR_NEAR = [{'seat_63'}, {'seat_64'}, {'seat_66'}]

    # A genuine traveling pattern that stays entirely on the OTHER side of
    # the room -- no cell adjacent to or matching seat_65.
    INVIGILATOR_FAR = [{'seat_60'}, {'seat_61'}, {'seat_63'}]

    # Sustained dip: 4 consecutive False at the end, timeout_frames=3 (reused
    # from _BRIEF_SPIKE_MAX_FRAMES) requires strictly >3.
    DIP_HISTORY = [True, True, False, False, False, False]
    NO_DIP_HISTORY = [True, True, True, True, True, True]

    def test_both_gates_true_tags(self):
        self.assertTrue(flag_seat_vacant_near_invigilator(
            self.DIP_HISTORY, self.INVIGILATOR_NEAR, self.SEAT_ID, self.CAMERA_ID))

    def test_invigilator_proximity_only_no_dip_does_not_tag(self):
        self.assertFalse(flag_seat_vacant_near_invigilator(
            self.NO_DIP_HISTORY, self.INVIGILATOR_NEAR, self.SEAT_ID, self.CAMERA_ID))

    def test_dip_only_no_invigilator_proximity_does_not_tag(self):
        # Single stationary spike, not a genuine traveling pattern --
        # is_invigilator_motion itself returns False (matches
        # test_single_stationary_cell_spike).
        stationary = [{'seat_1'}, {'seat_1'}, {'seat_1'}]
        self.assertFalse(flag_seat_vacant_near_invigilator(
            self.DIP_HISTORY, stationary, self.SEAT_ID, self.CAMERA_ID))

    def test_invigilator_at_non_adjacent_seat_does_not_tag(self):
        # is_invigilator_motion is True (genuine 3-cell travel), but none of
        # the touched cells are seat_65 or its one adjacent seat (seat_64).
        self.assertFalse(flag_seat_vacant_near_invigilator(
            self.DIP_HISTORY, self.INVIGILATOR_FAR, self.SEAT_ID, self.CAMERA_ID))


class TestBuildExplanation(unittest.TestCase):
    """
    §9 build_explanation(). Fixtures use the exact event schema
    P2P3Bridge._finalize_track() produces (seat_ids, activities, start_time/
    end_time, object_detected, metadata) -- same synthetic-fixture convention
    already used elsewhere in this file and in test_severity_scoring.py.
    Real-clip verification (DAHISAR1 clip 07 / Camera12 clip 4 events) is
    reported separately, not repeated here as unit tests.
    """

    def _event(self, **overrides):
        base = {
            "seat_ids": ["seat_14"],
            "activities": [],
            "start_time": 10.0,
            "end_time": 16.4,
            "object_detected": False,
        }
        base.update(overrides)
        return base

    def test_master_doc_example_exact(self):
        # "Seat 14, hand movement toward neighbor, 6.4s, phone detected 0.62"
        event = self._event(
            activities=["chit_passing:3+4"],
            object_detected=True,
            metadata=[{"frame": 100, "class": "phone", "confidence": 0.62}],
        )
        self.assertEqual(
            build_explanation(event),
            "Seat 14, hand movement toward neighbor, 6.4s, phone detected 0.62",
        )

    def test_motion_only_no_activities_no_object_still_valid(self):
        # Motion+fusion alone (no pose signal, no detector hit) must still
        # produce a valid, non-blank explanation -- never "None", never crash.
        event = self._event()
        self.assertEqual(build_explanation(event), "Seat 14, motion detected, 6.4s")

    def test_object_detected_false_omits_clause_entirely(self):
        event = self._event(object_detected=False, metadata=[{"frame": 1, "class": "phone", "confidence": 0.9}])
        result = build_explanation(event)
        self.assertNotIn("None", result)
        self.assertNotIn("phone", result)  # clause omitted even though metadata happens to be present

    def test_object_detected_true_but_no_metadata_omits_clause_not_crash(self):
        # Defensive case: object_detected=True but metadata missing/empty
        # (shouldn't happen given how _finalize_track sets both together, but
        # build_explanation must degrade gracefully, not raise, if it does).
        event = self._event(object_detected=True)
        result = build_explanation(event)
        self.assertEqual(result, "Seat 14, motion detected, 6.4s")

    def test_missing_confidence_omits_number_not_none(self):
        event = self._event(object_detected=True, metadata=[{"frame": 1, "class": "chit"}])
        result = build_explanation(event)
        self.assertIn("chit detected", result)
        self.assertNotIn("None", result)

    def test_picks_max_confidence_detection(self):
        event = self._event(object_detected=True, metadata=[
            {"frame": 1, "class": "phone", "confidence": 0.30},
            {"frame": 5, "class": "phone", "confidence": 0.81},
        ])
        self.assertIn("phone detected 0.81", build_explanation(event))

    def test_no_seat_id_falls_back_to_unknown(self):
        event = self._event(seat_ids=[])
        self.assertTrue(build_explanation(event).startswith("Seat unknown, "))

    def test_multi_seat_joined_with_slash(self):
        event = self._event(seat_ids=["seat_4", "seat_5"])
        self.assertTrue(build_explanation(event).startswith("Seat 4/5, "))

    def test_unmapped_activity_falls_back_to_readable_raw_string(self):
        event = self._event(activities=["some_future_signal"])
        self.assertIn("some future signal", build_explanation(event))

    def test_never_blank(self):
        # Minimal/degenerate event: no seat, no activities, no object.
        event = {"start_time": 0.0, "end_time": 0.0}
        result = build_explanation(event)
        self.assertTrue(result.strip())
        self.assertEqual(result, "Seat unknown, motion detected, 0.0s")


if __name__ == '__main__':
    unittest.main()
