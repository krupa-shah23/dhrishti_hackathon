import unittest
import numpy as np
from src.motion.baseline import RollingBaselineTracker

class TestRollingBaselineTracker(unittest.TestCase):
    def test_rolling_baseline_adaptation(self):
        # Initialize with baseline around 0.1
        initial_samples = {'seat_1': [0.1] * 100}
        tracker = RollingBaselineTracker(initial_samples, window_size=50)
        
        stats = tracker.get_stats()
        self.assertAlmostEqual(stats['seat_1'][0], 0.1)
        
        # Simulate a lighting change: intensity jumps to 0.8
        # We push 20 frames of 0.8. Window is 50, so 30 frames are 0.1, 20 are 0.8
        # Median should still be 0.1 (since 30 > 20)
        for _ in range(20):
            tracker.update({'seat_1': 0.8})
        
        stats = tracker.get_stats()
        self.assertAlmostEqual(stats['seat_1'][0], 0.1)
        
        # Push another 10 frames of 0.8. Now 20 frames of 0.1, 30 of 0.8.
        # Median should shift to 0.8!
        for _ in range(10):
            tracker.update({'seat_1': 0.8})
            
        stats = tracker.get_stats()
        self.assertAlmostEqual(stats['seat_1'][0], 0.8)
        
        # Confirm it recovers if it goes back to 0.1
        for _ in range(50):
            tracker.update({'seat_1': 0.1})
            
        stats = tracker.get_stats()
        self.assertAlmostEqual(stats['seat_1'][0], 0.1)

if __name__ == '__main__':
    unittest.main()
