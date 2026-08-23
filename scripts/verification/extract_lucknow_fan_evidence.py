import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.motion.motion import MotionEstimator, fuse_motion_signal

VIDEO = r"data\drishti\06_phone_use.mp4"
REGION = (860, 0, 1170, 110)
OUT = r"scratch_frames_dahisar1"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
SEEK_SEC = 2900.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(SEEK_SEC * fps))

estimator = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
x1, y1, x2, y2 = REGION

# Warm up to t=3000s (100s in), then grab 4 consecutive raw frames + masks
TARGET_START = 100.0 * fps  # frame offset from seek point
frame_i = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mag_map, mog2_mask = estimator.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mog2_mask)
    if frame_i >= TARGET_START and saved < 4:
        raw_crop = frame[y1:y2, x1:x2]
        mask_crop = fused[y1:y2, x1:x2]
        mask_bgr = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{OUT}/lucknow_fan_raw_{saved}.jpg", cv2.resize(raw_crop, None, fx=3, fy=3))
        cv2.imwrite(f"{OUT}/lucknow_fan_mask_{saved}.jpg", cv2.resize(mask_bgr, None, fx=3, fy=3))
        frac = float(np.count_nonzero(mask_crop)) / mask_crop.size
        print(f"saved={saved} frame_i={frame_i} t={SEEK_SEC + frame_i/fps:.2f}s frac={100*frac:.2f}%")
        saved += 1
    frame_i += 1
    if saved >= 4:
        break

cap.release()
print("done")
