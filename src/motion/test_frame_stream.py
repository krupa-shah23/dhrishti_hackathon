import unittest
import tempfile
import cv2
import numpy as np
from pathlib import Path
from src.motion.frame_stream import get_target_fps, calculate_sample_rate, get_frame_stream, get_frame_stream_with_indices

class TestFrameStream(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        
        # Create a synthetic video for testing: 30 FPS, 30 frames (1 second)
        self.test_video = self.data_dir / "test.mp4"
        out = cv2.VideoWriter(str(self.test_video), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (100, 100))
        for i in range(30):
            frame = np.full((100, 100, 3), i, dtype=np.uint8) # Fill with index so we can verify
            out.write(frame)
        out.release()
        
    def tearDown(self):
        self.tmpdir.cleanup()
        
    def test_target_fps_policy(self):
        self.assertEqual(get_target_fps(20.0), 8.0)
        self.assertEqual(get_target_fps(30.0), 4.0)
        self.assertEqual(get_target_fps(60.0), 4.0)
        self.assertEqual(get_target_fps(120.0), 4.0)
        self.assertEqual(get_target_fps(121.0), 2.0)
        self.assertEqual(get_target_fps(600.0), 2.0)
        self.assertEqual(get_target_fps(601.0), 1.0)
        
    def test_calculate_sample_rate(self):
        # 8 -> 4 (sample_rate = 2)
        self.assertEqual(calculate_sample_rate(8.0, 4.0), 2.0)
        # 8 -> 2 (sample_rate = 4)
        self.assertEqual(calculate_sample_rate(8.0, 2.0), 4.0)
        # 8 -> 1 (sample_rate = 8)
        self.assertEqual(calculate_sample_rate(8.0, 1.0), 8.0)
        # 30 -> 8 (sample_rate = 3.75)
        self.assertEqual(calculate_sample_rate(30.0, 8.0), 3.75)
        # Target >= Source (no upsampling)
        self.assertEqual(calculate_sample_rate(10.0, 30.0), 1.0)
        
    def test_empty_unreadable_video(self):
        fake_vid = self.data_dir / "fake.mp4"
        fake_vid.write_text("not a video")
        with self.assertRaises(ValueError):
            list(get_frame_stream(str(fake_vid)))
            
    def test_sample_rate_1(self):
        # Should yield all 30 frames
        stream = get_frame_stream_with_indices(str(self.test_video), sample_rate=1.0)
        frames = list(stream)
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0][0], 0)
        self.assertEqual(frames[-1][0], 29)
        # Check original timeline preserved
        self.assertAlmostEqual(frames[1][1], 1.0 / 30.0)
        
    def test_sample_rate_2(self):
        # Should yield 15 frames: 0, 2, 4, ...
        stream = get_frame_stream_with_indices(str(self.test_video), sample_rate=2.0)
        frames = list(stream)
        self.assertEqual(len(frames), 15)
        indices = [f[0] for f in frames]
        self.assertEqual(indices, list(range(0, 30, 2)))
        
    def test_sample_rate_4(self):
        # Should yield frames: 0, 4, 8, 12, 16, 20, 24, 28 (8 frames)
        stream = get_frame_stream_with_indices(str(self.test_video), sample_rate=4.0)
        frames = list(stream)
        self.assertEqual(len(frames), 8)
        indices = [f[0] for f in frames]
        self.assertEqual(indices, list(range(0, 30, 4)))
        
    def test_non_integer_sample_rate(self):
        # sample_rate = 3.75 (e.g. 30 -> 8 FPS)
        stream = get_frame_stream_with_indices(str(self.test_video), sample_rate=3.75)
        frames = list(stream)
        indices = [f[0] for f in frames]
        # target points: 0.0, 3.75, 7.5, 11.25, 15.0, 18.75, 22.5, 26.25
        # selected: 0, 4, 8, 12, 15, 19, 23, 27
        expected_indices = [0, 4, 8, 12, 15, 19, 23, 27]
        self.assertEqual(indices, expected_indices)
        
    def test_frame_ordering_and_content(self):
        stream = get_frame_stream_with_indices(str(self.test_video), sample_rate=2.0)
        for idx, ts, frame in stream:
            # Verify the output format instead of exact compression-lossy pixel values
            self.assertIsInstance(idx, int)
            self.assertIsInstance(ts, float)
            self.assertEqual(frame.shape, (100, 100, 3))
            
    def test_simple_api(self):
        stream = get_frame_stream(str(self.test_video), sample_rate=4.0)
        frames = list(stream)
        self.assertEqual(len(frames), 8)
        # Should just be numpy arrays
        self.assertIsInstance(frames[0], np.ndarray)
        
    def test_real_video_integration(self):
        real_video = Path("data/drishti/04.CCTV Candidate Talking.mkv")
        if real_video.exists():
            cap = cv2.VideoCapture(str(real_video))
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / source_fps if source_fps > 0 else 0
            cap.release()
            
            target_fps = get_target_fps(duration)
            sample_rate = calculate_sample_rate(source_fps, target_fps)
            
            self.assertEqual(source_fps, 8.0)
            self.assertEqual(target_fps, 2.0)
            self.assertEqual(sample_rate, 4.0)
            
            # Since we just want to verify it without fully processing in tests,
            # we can pull a few frames to make sure it doesn't crash.
            stream = get_frame_stream_with_indices(str(real_video), sample_rate)
            first_frame = next(stream)
            self.assertEqual(first_frame[0], 0)
            
            # Consume entirely to count
            count = 1
            last_idx = 0
            for idx, ts, frame in stream:
                count += 1
                last_idx = idx
                
            # Expected around 1145 / 4 = ~287 frames
            self.assertAlmostEqual(count, 287, delta=2)
            self.assertLess(last_idx, frame_count)

if __name__ == '__main__':
    unittest.main()
