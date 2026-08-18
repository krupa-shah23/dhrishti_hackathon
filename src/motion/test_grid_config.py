import unittest
import numpy as np
from src.motion.grid_config import get_grid_config, validate_grid_config, get_seat_mask, GRID_CONFIGS

class TestGridConfig(unittest.TestCase):
    def test_camera12_exists(self):
        config = get_grid_config("Camera12")
        self.assertIn("resolution", config)
        self.assertIn("seats", config)
        self.assertEqual(config["resolution"], (640, 480))
        
    def test_unknown_camera_returns_empty(self):
        config = get_grid_config("UnknownCamera")
        self.assertEqual(config, {})
        
    def test_seats_have_valid_format(self):
        config = get_grid_config("Camera12")
        errors = validate_grid_config(config)
        self.assertEqual(errors, [])
        
    def test_unique_seat_ids(self):
        # Dict inherently enforces unique keys, but let's check our manual definition
        seats = GRID_CONFIGS["Camera12"]["seats"]
        self.assertEqual(len(seats), len(set(seats.keys())))
        
    def test_resolution_scaling(self):
        # 1280x960 is exactly 2x 640x480
        config = get_grid_config("Camera12", 1280, 960)
        self.assertEqual(config["resolution"], (1280, 960))
        
        orig = get_grid_config("Camera12")
        for seat_id, box in orig["seats"].items():
            scaled_box = config["seats"][seat_id]
            self.assertEqual(scaled_box[0], box[0] * 2)
            self.assertEqual(scaled_box[1], box[1] * 2)
            self.assertEqual(scaled_box[2], box[2] * 2)
            self.assertEqual(scaled_box[3], box[3] * 2)
            
    def test_overlap_validation_custom(self):
        # Create a bad config to test validation
        bad_config = {
            "resolution": (100, 100),
            "seats": {
                "seat_0": (50, 50, 10, 10),    # negative area
                "seat_1": (-10, 0, 50, 50),    # out of bounds
                "seat_2": (0, 0, 200, 50),     # exceeds width
            }
        }
        errors = validate_grid_config(bad_config)
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("negative area" in e for e in errors))
        self.assertTrue(any("out of frame bounds" in e for e in errors))
        
    def test_configuration_not_mutated(self):
        orig_config = GRID_CONFIGS["Camera12"]["seats"].copy()
        config = get_grid_config("Camera12", 1280, 960)
        # Verify original wasn't changed
        self.assertEqual(GRID_CONFIGS["Camera12"]["seats"], orig_config)
        
    def test_seat_mask(self):
        mask = get_seat_mask((480, 640, 3), "seat_66", "Camera12")
        self.assertEqual(mask.shape, (480, 640))
        # Should be binary 0/255
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
        
        # Check specific bounds
        orig = get_grid_config("Camera12")["seats"]["seat_66"]
        x1, y1, x2, y2 = orig
        self.assertEqual(mask[y1+1, x1+1], 255)
        
if __name__ == '__main__':
    unittest.main()
