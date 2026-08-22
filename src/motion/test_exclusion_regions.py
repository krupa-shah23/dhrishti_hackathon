import unittest
import numpy as np
import cv2
from src.motion.exclusion_regions import get_exclusion_mask, CAMERA_CONFIGS
from src.motion.motion import MotionEstimator

class TestExclusionRegions(unittest.TestCase):
    def test_camera12_returns_valid_mask(self):
        mask = get_exclusion_mask("Camera12", 640, 480)
        self.assertEqual(mask.shape, (480, 640))
        self.assertEqual(mask.dtype, np.uint8)
        
    def test_unknown_camera_returns_all_valid(self):
        mask = get_exclusion_mask("UnknownCamera", 640, 480)
        self.assertTrue(np.all(mask == 255))
        
    def test_camera12_640x480_timestamp_excluded(self):
        mask = get_exclusion_mask("Camera12", 640, 480)
        # Check inside timestamp rect (30, 0, 295, 125)
        self.assertEqual(mask[60, 150], 0)
        # Check outside
        self.assertEqual(mask[200, 150], 255)
        
    def test_camera12_640x480_overlay_excluded(self):
        mask = get_exclusion_mask("Camera12", 640, 480)
        # Check inside Camera ID rect (410, 355, 525, 450)
        self.assertEqual(mask[400, 450], 0)

    def test_camera12_1280x720_overlays_excluded(self):
        mask = get_exclusion_mask("Camera12", 1280, 720)
        # Check inside top-left rect (43, 0, 750, 115)
        self.assertEqual(mask[50, 200], 0)
        # Check inside big bottom-right (951, 607, 1227, 676)
        self.assertEqual(mask[650, 1000], 0)
        # Check inside small bottom-right (1150, 696, 1280, 720)
        self.assertEqual(mask[710, 1200], 0)
        
    def test_candidate_regions_not_excluded(self):
        mask = get_exclusion_mask("Camera12", 640, 480)
        # Check middle of screen where candidates sit
        self.assertEqual(mask[240, 320], 255)
        
    def test_missing_resolution_returns_all_valid(self):
        # 1280x960 is not in the config for Camera12
        mask = get_exclusion_mask("Camera12", 1280, 960)
        self.assertEqual(mask.shape, (960, 1280))
        self.assertTrue(np.all(mask == 255))
        
    def test_invalid_resolution_handled_safely(self):
        # Extremely small
        mask = get_exclusion_mask("Camera12", 10, 10)
        self.assertEqual(mask.shape, (10, 10))
        # Zero width/height shouldn't crash
        mask_zero = get_exclusion_mask("Camera12", 0, 0)
        self.assertEqual(mask_zero.shape, (0, 0))
        
    def test_configuration_mutation(self):
        # Ensure the getter doesn't mutate config
        orig_config = CAMERA_CONFIGS[("Camera12", (640, 480))].copy()
        mask1 = get_exclusion_mask("Camera12", 640, 480)
        self.assertEqual(CAMERA_CONFIGS[("Camera12", (640, 480))], orig_config)
        
    def test_motion_estimator_integration(self):
        mask = get_exclusion_mask("Camera12", 640, 480)
        estimator = MotionEstimator(camera_mask=mask)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mag_map, out_mask = estimator.get_motion_mask(frame)
        self.assertEqual(out_mask.shape, (480, 640))
        
if __name__ == '__main__':
    unittest.main()
