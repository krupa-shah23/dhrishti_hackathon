"""
verify_fan_region_motion.py
§B item 3 (fans/curtains/vents, connects to calibrate_baseline redundancy question).

01_phone_use.mkv (Camera04) has a ceiling fan visible in the top-left corner of frame
(~pixels (0,0)-(400,130) at native 1280x720). Runs MOG2 continuously over the whole
clip (131s @ 25fps) and measures per-frame foreground-pixel fraction inside the fan's
crop region only, to check: does periodic fan motion (if the fan is running) produce
sustained false motion above motion_threshold=0.05, or does MOG2 absorb it into
background within a reasonable window (as production already relies on, with no
rolling baseline)?
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.motion.motion import MotionEstimator, fuse_motion_signal

VIDEO = r"data\drishti\01_phone_use.mkv"
FAN_REGION = (0, 0, 400, 130)  # x1,y1,x2,y2 native 1280x720, from visual inspection

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps}, total_frames={total}, duration={total/fps:.1f}s")

estimator = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
x1, y1, x2, y2 = FAN_REGION
records = []  # (t, fan_fraction)

for fi in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    mag_map, mog2_mask = estimator.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mog2_mask)
    crop = fused[y1:y2, x1:x2]
    frac = float(np.count_nonzero(crop)) / crop.size if crop.size else 0.0
    records.append((fi / fps, frac))

cap.release()

vals = np.array([r[1] for r in records])
print(f"\nFan-region motion fraction over {len(records)} frames:")
print(f"  overall mean={100*vals.mean():.3f}%  max={100*vals.max():.3f}%  std={100*vals.std():.3f}%")

# Bucket into 10s windows to see if there's sustained (not just transient) false motion,
# and whether it decays (MOG2 absorbing it) or stays elevated.
WINDOW = 10.0
n_windows = int(total / fps / WINDOW) + 1
print(f"\nPer-{WINDOW:.0f}s-window fan-region mean/max (threshold=5.0% for reference):")
for w in range(n_windows):
    t0, t1 = w * WINDOW, (w + 1) * WINDOW
    wv = [f for t, f in records if t0 <= t < t1]
    if not wv:
        continue
    wv = np.array(wv)
    flag = " <-- ABOVE 5%" if wv.mean() * 100 >= 5.0 else ""
    print(f"  [{t0:5.0f}-{t1:5.0f}s] mean={100*wv.mean():.3f}%  max={100*wv.max():.3f}%{flag}")

# Frames where fan-region alone would cross the production motion_threshold=0.05
above = [(t, f) for t, f in records if f >= 0.05]
print(f"\nFrames where fan-region ALONE >= 5% (motion_threshold): {len(above)} / {len(records)}")
if above:
    for t, f in above[:20]:
        print(f"    t={t:.2f}s  frac={100*f:.2f}%")
