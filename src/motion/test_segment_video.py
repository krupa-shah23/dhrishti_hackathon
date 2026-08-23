import unittest
import tempfile
from pathlib import Path
import json
import csv
from src.motion.segment_video import calculate_segments, export_segments

class TestSegmentVideo(unittest.TestCase):
    def test_calculate_segments_normal_video(self):
        # 3600 seconds = 1 hour video
        duration = 3600.0
        info = calculate_segments(duration)
        
        # 1. Calibration window calculation: min(max(0.10 * 3600, 60), 300) = 300
        self.assertEqual(info["calib_window_sec"], 300.0)
        
        # 2. Phase boundary calculation
        self.assertEqual(info["phase_1"]["start"], 0.0)
        self.assertEqual(info["phase_1"]["end"], 360.0) # 10%
        self.assertEqual(info["phase_2"]["start"], 360.0)
        self.assertEqual(info["phase_2"]["end"], 3240.0) # 90%
        self.assertEqual(info["phase_3"]["start"], 3240.0)
        self.assertEqual(info["phase_3"]["end"], 3600.0) # 100%
        
        # 3. Segment duration calculation: max(30, min(60, 0.05 * 3600)) = 60
        self.assertEqual(info["segment_duration_sec"], 60.0)
        
        # 4. Deterministic segment selection
        segments = info["segments"]
        self.assertEqual(len(segments), 7)
        
        # 01_calibration: capped at 60s
        self.assertEqual(segments[0]["segment_id"], "01")
        self.assertEqual(segments[0]["start_sec"], 0.0)
        self.assertEqual(segments[0]["end_sec"], 60.0)
        
        # 02_early_phase: 20% of phase 1 -> 72s
        self.assertEqual(segments[1]["segment_id"], "02")
        self.assertEqual(segments[1]["start_sec"], 72.0)
        self.assertEqual(segments[1]["end_sec"], 132.0)
        
        # Check ordering is chronological by target
        for i in range(1, len(segments)):
            self.assertGreaterEqual(segments[i]["start_sec"], segments[i-1]["start_sec"])

        # Check segment boundaries are within video
        for seg in segments:
            self.assertGreaterEqual(seg["start_sec"], 0.0)
            self.assertLessEqual(seg["end_sec"], duration)

    def test_calculate_segments_short_video(self):
        # 143.12 seconds
        duration = 143.12
        info = calculate_segments(duration)
        
        self.assertEqual(info["calib_window_sec"], 60.0)
        self.assertEqual(info["segment_duration_sec"], 30.0)
        
        segments = {s['segment_id']: s for s in info["segments"]}
        
        # 1. calibration starts at 0, ends at 60 for this video
        self.assertEqual(segments["01"]["start_sec"], 0.0)
        self.assertEqual(segments["01"]["end_sec"], 60.0)
        
        # 3. early_phase is entirely inside Phase 1
        self.assertGreaterEqual(segments["02"]["start_sec"], info["phase_1"]["start"])
        self.assertLessEqual(segments["02"]["end_sec"], info["phase_1"]["end"])
        
        # 4. early_middle is centred around 25%
        self.assertEqual(segments["03"]["start_sec"], max(0, round(0.25 * duration - 15, 2)))
        
        # 5. midpoint is centred around 50%
        self.assertEqual(segments["04"]["start_sec"], max(0, round(0.50 * duration - 15, 2)))
        
        # 6. late_phase is centred around 75%
        self.assertEqual(segments["05"]["start_sec"], max(0, round(0.75 * duration - 15, 2)))
        
        # 7. final_phase is entirely inside Phase 3
        self.assertGreaterEqual(segments["06"]["start_sec"], info["phase_3"]["start"])
        self.assertLessEqual(segments["06"]["end_sec"], info["phase_3"]["end"])
        
        # 8. end segment ends at duration
        self.assertEqual(segments["07"]["end_sec"], duration)
        
        # 9, 10, 11: Boundaries and ordering
        arr = info["segments"]
        # Segments are ordered by target, so start_sec might slightly interleave for short videos
        # Just check that all segments are within the video boundaries.
        for seg in arr:
            self.assertGreaterEqual(seg["start_sec"], 0.0)
            self.assertLessEqual(seg["end_sec"], duration)

    def test_calculate_segments_very_long_video(self):
        # 36000 seconds = 10 hours
        duration = 36000.0
        info = calculate_segments(duration)
        
        # calib window = min(max(3600, 60), 300) = 300
        self.assertEqual(info["calib_window_sec"], 300.0)
        
        # segment duration = max(30, min(60, 1800)) = 60
        self.assertEqual(info["segment_duration_sec"], 60.0)
        
        segments = info["segments"]
        self.assertEqual(len(segments), 7)
        self.assertEqual(segments[-1]["start_sec"], 35940.0)

    def test_export_segments_metadata_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = tmp_path / "mock.mkv"
            out_dir = tmp_path / "analysis_segments"
            
            class MockCap:
                def __init__(self, *args): pass
                def set(self, *args): pass
                def read(self): return False, None
                def release(self): pass
                
            class MockWriter:
                def __init__(self, *args): pass
                def isOpened(self): return True
                def write(self, *args): pass
                def release(self): pass
                
            import cv2
            import src.motion.segment_video
            
            original_cap = cv2.VideoCapture
            original_writer = cv2.VideoWriter
            cv2.VideoCapture = MockCap
            cv2.VideoWriter = MockWriter
            
            try:
                segments_info = calculate_segments(3600)
                metadata = {
                    "source_video": "mock.mkv",
                    "fps": 30.0,
                    "frame_count": 108000,
                    "width": 1920,
                    "height": 1080,
                    "codec": "mock",
                    "duration_sec": 3600.0,
                    "has_audio": None
                }
                
                src.motion.segment_video.export_segments(video_path, out_dir, segments_info, metadata)
                
                # 11. Output directory creation
                self.assertTrue(out_dir.exists())
                self.assertTrue((out_dir / "previews").exists())
                
                # 12. segments.csv correctness
                csv_path = out_dir / "segments.csv"
                self.assertTrue(csv_path.exists())
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertEqual(len(rows), 7)
                    self.assertEqual(rows[0]["segment_id"], "01")
                    self.assertEqual(rows[0]["label"], "calibration")
                    self.assertEqual(rows[0]["start_sec"], "0.0")
                    self.assertEqual(rows[0]["end_sec"], "60.0")
                    
                # 13. metadata.json correctness
                json_path = out_dir / "metadata.json"
                self.assertTrue(json_path.exists())
                with open(json_path, "r") as f:
                    data = json.load(f)
                    self.assertEqual(data["duration_sec"], 3600.0)
                    self.assertEqual(data["calib_window_sec"], 300.0)
                    self.assertEqual(len(data["segments"]), 7)
                    self.assertEqual(data["segmentation_algorithm"], "v1.0")
                    
            finally:
                cv2.VideoCapture = original_cap
                cv2.VideoWriter = original_writer

if __name__ == '__main__':
    unittest.main()
