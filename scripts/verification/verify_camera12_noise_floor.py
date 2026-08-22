"""
verify_camera12_noise_floor.py
Camera12 equivalent of the DAHISAR1 per-seat MOG2 noise-floor investigation
(calibrate_baseline redundancy check), for §B item 1 (near/far adaptive baseline).

Uses 03_mobile_usage.mkv -- the only Camera12 clip long enough (281.79s) to support
a real warmup+measure split (04_candidate_talking.mkv is only 143s).

Same methodology as verify_mog2_noise_floor_post_warmup.py (DAHISAR1): let MOG2 run
continuously through a warmup window, then measure per-seat _seat_intensity (baseline.py's
seat-local normalized fraction) over a following measurement window.
"""
import sys
sys.path.insert(0, ".")

from src.motion.frame_stream import get_frame_stream_with_indices, calculate_sample_rate
from src.motion.grid_config import get_grid_config
from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.baseline import _seat_intensity
import cv2

VIDEO = r"data\drishti\03_mobile_usage.mkv"
CAMERA_ID = "Camera12"
WARMUP_SEC = 150.0
MEASURE_SEC = 120.0
END_SEC = WARMUP_SEC + MEASURE_SEC

cap = cv2.VideoCapture(VIDEO)
source_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

sample_rate = calculate_sample_rate(source_fps, target_fps=2.0)
print(f"source_fps={source_fps}, sample_rate={sample_rate}")

target_h = 480
target_w = int(width * (target_h / height)) if height > 0 else width
grid = get_grid_config(CAMERA_ID, target_w, target_h)
seats = grid["seats"]
print(f"seats: {list(seats.keys())}")
for sid, (x1, y1, x2, y2) in seats.items():
    print(f"  {sid}: area={(x2-x1)*(y2-y1)}")

estimator = MotionEstimator()
values = {sid: [] for sid in seats}
n_measured = 0

for frame_idx, ts, frame in get_frame_stream_with_indices(VIDEO, camera_id=CAMERA_ID, sample_rate=sample_rate):
    if ts > END_SEC:
        break
    mag_map, mog2_mask = estimator.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mog2_mask)
    if ts >= WARMUP_SEC:
        n_measured += 1
        for sid, (x1, y1, x2, y2) in seats.items():
            values[sid].append(_seat_intensity(fused, x1, y1, x2, y2))

print(f"\nn_measured={n_measured}")
print(f"\nPer-seat noise floor AFTER {WARMUP_SEC:.0f}s of MOG2 warm-up, over the following {MEASURE_SEC:.0f}s:")
FRONT_ROW = {"seat_60", "seat_61"}
BACK_ROW = {"seat_63", "seat_64", "seat_65"}
for sid in sorted(values):
    v = values[sid]
    if not v:
        print(f"  {sid}: no samples")
        continue
    mean_pct = 100.0 * sum(v) / len(v)
    max_pct = 100.0 * max(v)
    row = "FRONT(near)" if sid in FRONT_ROW else ("BACK(far)" if sid in BACK_ROW else "seat_66(side/near)")
    status = "ACTIVE (>=5%)" if mean_pct >= 5.0 else "IDLE (<5%)"
    print(f"  {sid} [{row}]: mean={mean_pct:.2f}%  max={max_pct:.2f}%  n={len(v)}  {status}")
