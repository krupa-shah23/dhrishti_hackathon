import unittest
from src.motion.heuristics import is_invigilator_motion, tag_interventions, is_seat_vacated

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

if __name__ == '__main__':
    unittest.main()
