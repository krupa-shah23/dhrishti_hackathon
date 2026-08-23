"""
verify_lucknow_fan_vent.py
§B item 3, second real clip: LUCKNOW1 (06_phone_use.mp4) has a wall-mounted ceiling fan
+ split-AC vent unit visible at native-coords bbox ~(860,0,1170,110) (top-right of frame,
confirmed via visual inspection at t=3000s). Seeks near t=2900s, lets MOG2 warm up for a
while, then measures per-frame foreground fraction in that region over several minutes to
check for sustained periodic false motion vs MOG2 absorption.
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.motion.motion import MotionEstimator, fuse_motion_signal

VIDEO = r"data\drishti\06_phone_use.mp4"
FAN_VENT_REGION = (860, 0, 1170, 110)  # native 1280x720

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
SEEK_SEC = 2900.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(SEEK_SEC * fps))

estimator = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
x1, y1, x2, y2 = FAN_VENT_REGION
records = []

N_FRAMES = int(400 * fps)  # ~400s of video time
for i in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
        print(f"video ended at i={i}")
        break
    mag_map, mog2_mask = estimator.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mog2_mask)
    crop = fused[y1:y2, x1:x2]
    frac = float(np.count_nonzero(crop)) / crop.size if crop.size else 0.0
    records.append((i / fps, frac))

cap.release()
print(f"Collected {len(records)} frames starting near t={SEEK_SEC}s")

WARMUP_DISCARD = 60.0  # discard first 60s as MOG2 cold-start after the seek
measured = [(t, f) for t, f in records if t >= WARMUP_DISCARD]
vals = np.array([f for _, f in measured])
print(f"\nAfter discarding first {WARMUP_DISCARD:.0f}s (cold-start post-seek):")
print(f"  n={len(vals)}  mean={100*vals.mean():.3f}%  max={100*vals.max():.3f}%  std={100*vals.std():.3f}%")

WINDOW = 20.0
t_min = measured[0][0] if measured else 0
n_windows = int((N_FRAMES / fps - t_min) / WINDOW) + 1
print(f"\nPer-{WINDOW:.0f}s-window fan/vent-region mean/max:")
for w in range(n_windows):
    t0, t1 = t_min + w * WINDOW, t_min + (w + 1) * WINDOW
    wv = [f for t, f in measured if t0 <= t < t1]
    if not wv:
        continue
    wv = np.array(wv)
    flag = " <-- ABOVE 5%" if wv.mean() * 100 >= 5.0 else ""
    print(f"  [{t0:6.0f}-{t1:6.0f}s] mean={100*wv.mean():.3f}%  max={100*wv.max():.3f}%{flag}")

above = [(t, f) for t, f in measured if f >= 0.05]
print(f"\nFrames where fan/vent-region ALONE >= 5% (motion_threshold): {len(above)} / {len(measured)}")
for t, f in above[:20]:
    print(f"    t={t:.2f}s  frac={100*f:.2f}%")
