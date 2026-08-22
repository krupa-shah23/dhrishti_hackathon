import unittest
import tempfile
import pandas as pd
from pathlib import Path
from src.motion.build_manifest import build_manifest, CONFIG

class TestBuildManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        
    def tearDown(self):
        self.tmpdir.cleanup()
        
    def test_empty_directory(self):
        df = build_manifest(self.data_dir)
        self.assertTrue(df.empty)
        # Check mandatory columns exist even if empty
        mandatory = ["filename", "duration_sec", "resolution", "fps", "has_audio", "camera_id", "near_empty_start"]
        for col in mandatory:
            self.assertIn(col, df.columns)
            
    def test_no_supported_video_files(self):
        (self.data_dir / "test.txt").write_text("not a video")
        df = build_manifest(self.data_dir)
        self.assertTrue(df.empty)

    def test_unsupported_extensions_ignored(self):
        (self.data_dir / "test.mp3").write_text("audio only")
        df = build_manifest(self.data_dir)
        self.assertTrue(df.empty)

    def test_unreadable_corrupt_video(self):
        video_path = self.data_dir / "corrupt.mp4"
        video_path.write_text("corrupt data")
        
        # It should not crash, but just log failure and skip
        df = build_manifest(self.data_dir)
        self.assertTrue(df.empty)

    def test_camera_id_and_near_empty_configuration(self):
        import cv2
        # Mock cv2.VideoCapture to pretend video is readable
        class MockCap:
            def __init__(self, *args): pass
            def isOpened(self): return True
            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS: return 30.0
                if prop == cv2.CAP_PROP_FRAME_COUNT: return 300
                if prop == cv2.CAP_PROP_FRAME_WIDTH: return 1280
                if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 720
                if prop == cv2.CAP_PROP_FOURCC: return 828601953 # 'avc1' or something
                return 0
            def release(self): pass
            
        original_cap = cv2.VideoCapture
        cv2.VideoCapture = MockCap
        
        try:
            video_path = self.data_dir / "04_candidate_talking.mkv"
            video_path.write_text("fake valid video data")
            
            df = build_manifest(self.data_dir)
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            
            self.assertEqual(row["filename"], "04_candidate_talking.mkv")
            self.assertEqual(row["camera_id"], "Camera12")
            self.assertEqual(row["near_empty_start"], False)
            self.assertEqual(row["duration_sec"], 10.0)
            self.assertEqual(row["calib_window_sec"], 60.0) # min(max(0.1*10, 60), 300) = 60
            self.assertEqual(row["resolution"], "1280x720")
            
            # test another video without config
            other_video = self.data_dir / "unknown.mp4"
            other_video.write_text("data")
            df2 = build_manifest(self.data_dir)
            self.assertEqual(len(df2), 2)
            unknown_row = df2[df2['filename'] == 'unknown.mp4'].iloc[0]
            self.assertTrue(pd.isna(unknown_row["camera_id"]) or unknown_row["camera_id"] is None)
            self.assertTrue(pd.isna(unknown_row["near_empty_start"]) or unknown_row["near_empty_start"] is None)
            
        finally:
            cv2.VideoCapture = original_cap

    def test_excluded_analysis_segments(self):
        import cv2
        class MockCap:
            def __init__(self, *args): pass
            def isOpened(self): return True
            def get(self, prop): return 30.0
            def release(self): pass
            
        original_cap = cv2.VideoCapture
        cv2.VideoCapture = MockCap
        
        try:
            # Create normal video
            normal_vid = self.data_dir / "normal.mp4"
            normal_vid.write_text("data")
            
            # Create excluded video
            analysis_dir = self.data_dir / "analysis_segments"
            analysis_dir.mkdir()
            excluded_vid = analysis_dir / "01_calibration.mp4"
            excluded_vid.write_text("data")
            
            # Create nested unrelated directory
            nested_dir = self.data_dir / "other_nested_dir"
            nested_dir.mkdir()
            nested_vid = nested_dir / "nested.mp4"
            nested_vid.write_text("data")
            
            df = build_manifest(self.data_dir)
            # Should discover normal.mp4 and nested.mp4, but NOT 01_calibration.mp4
            self.assertEqual(len(df), 2)
            filenames = df["filename"].tolist()
            self.assertIn("normal.mp4", filenames)
            self.assertIn("nested.mp4", filenames)
            self.assertNotIn("01_calibration.mp4", filenames)
        finally:
            cv2.VideoCapture = original_cap

    def test_real_drishti_dataset(self):
        # Integration test for the real dataset
        real_data_dir = Path("data/drishti")
        if real_data_dir.exists() and (real_data_dir / "04.CCTV Candidate Talking.mkv").exists():
            df = build_manifest(real_data_dir)
            # The real DRISHTI dataset produces exactly ONE manifest row.
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["filename"], "04.CCTV Candidate Talking.mkv")
            self.assertEqual(row["camera_id"], "Camera12")
            self.assertEqual(row["near_empty_start"], False)

if __name__ == '__main__':
    unittest.main()
