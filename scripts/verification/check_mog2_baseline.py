"""
check_mog2_baseline.py
Verifies whether MOG2 alone successfully absorbs background noise (like fans/lighting drift)
on real clips, or if it leaks continuous noise that a per-seat baseline would have been needed to catch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')
import cv2
import numpy as np
from collections import defaultdict
from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.grid_config import get_grid_config
from src.motion.baseline import _seat_intensity

def check_clip(video_path, camera_id):
    print(f"\n--- Checking MOG2 background absorption on {camera_id} ---")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    grid = get_grid_config(camera_id, w, h)
    seats = grid.get('seats', {})
    if not seats:
        print("No seats configured.")
        return

    estimator = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
    
    # Track the peak foreground fraction per seat after the 60-second warmup period
    warmup_frames = int(fps * min(60, (cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps)*0.5))
    peak_intensities = defaultdict(float)
    avg_intensities = defaultdict(list)
    
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mag_map, mog2_mask = estimator.get_motion_mask(frame)
        fused = fuse_motion_signal(mag_map, mog2_mask)
        
        # Only measure noise *after* MOG2 has had time to build its baseline
        if fi > warmup_frames:
            for sid, (x1, y1, x2, y2) in seats.items():
                intensity = _seat_intensity(fused, x1, y1, x2, y2)
                avg_intensities[sid].append(intensity)
                if intensity > peak_intensities[sid]:
                    peak_intensities[sid] = intensity
        
        if fi > int(fps * 120):
            break
                    
        fi += 1
        if fi % 1000 == 0:
            print(f"  Processed {fi} frames...")

    cap.release()
    print("MOG2 Post-Warmup Background Noise (Fraction of Seat Area):")
    for sid in sorted(seats.keys()):
        if avg_intensities[sid]:
            mean_val = np.mean(avg_intensities[sid])
            peak_val = peak_intensities[sid]
            print(f"  {sid:10s} : Mean={mean_val:.4f}, Peak={peak_val:.4f}")
        else:
            print(f"  {sid:10s} : Clip shorter than warmup.")

if __name__ == '__main__':
    check_clip('data/drishti/07_seat_exchange.mkv', 'DAHISAR1')
