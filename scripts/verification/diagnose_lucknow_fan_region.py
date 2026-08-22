"""
Root-cause diagnosis for the LUCKNOW1 fan/vent-region sustained MOG2 foreground
(§B item 3 follow-up). Ffmpeg -v warning decode over the measured window
(t=2800-3400s) reported ZERO decode errors -- codec/compression damage is ruled
out. This script checks the remaining candidates:

1. Is the elevated reading a PERMANENT characteristic of this pixel region
   (present at an unrelated, early timestamp too), or specific to t~2900-3300s?
2. Pixel-level frame-to-frame diff: magnitude and spatial pattern in the
   fan/vent region vs. a same-frame STATIC control region (plain wall).
3. Whether the diff pattern is a coherent moving edge (real motion, e.g.
   blade/panel vibration) or scattered per-pixel jitter (sensor/exposure noise).
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

VIDEO = r"data\drishti\06_phone_use.mp4"
FAN_REGION = (860, 0, 1170, 110)       # fixture region (fan + AC vent + noticeboard)
CONTROL_REGION = (0, 200, 300, 400)    # plain wall/locker area, same frame, no fixture

def measure_window(seek_sec, n_seconds, label):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_sec * fps))
    n_frames = int(n_seconds * fps)

    prev = None
    fan_diffs = []
    ctrl_diffs = []
    fx1, fy1, fx2, fy2 = FAN_REGION
    cx1, cy1, cx2, cy2 = CONTROL_REGION

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            fan_crop = diff[fy1:fy2, fx1:fx2]
            ctrl_crop = diff[cy1:cy2, cx1:cx2]
            fan_diffs.append(fan_crop)
            ctrl_diffs.append(np.mean(ctrl_crop))
        prev = gray
    cap.release()

    if not fan_diffs:
        print(f"[{label}] no frames collected")
        return

    fan_stack = np.stack(fan_diffs)  # (N, H, W)
    mean_diff_per_pixel = fan_stack.mean(axis=0)  # average over time, per pixel
    overall_fan_mean = fan_stack.mean()
    overall_ctrl_mean = np.mean(ctrl_diffs)

    # fraction of pixels with meaningfully nonzero avg diff (edge-like vs uniform)
    active_px_frac = float(np.mean(mean_diff_per_pixel > 3.0))

    print(f"\n[{label}] seek={seek_sec}s, n_frame_pairs={len(fan_diffs)}")
    print(f"  fan-region  frame-to-frame |diff| mean={overall_fan_mean:.2f}  max_pixel_mean={mean_diff_per_pixel.max():.2f}")
    print(f"  control-region (plain wall) frame-to-frame |diff| mean={overall_ctrl_mean:.2f}")
    print(f"  fan-region: fraction of pixels with time-avg |diff|>3.0 (i.e. NOT random single-pixel noise): {100*active_px_frac:.1f}%")

    return mean_diff_per_pixel, overall_fan_mean, overall_ctrl_mean


print("=== Test 1: is this permanent (early, unrelated timestamp) or window-specific? ===")
measure_window(100.0, 20.0, "EARLY t=100-120s")
measure_window(2960.0, 20.0, "FLAGGED-WINDOW t=2960-2980s")
measure_window(5000.0, 20.0, "LATE t=5000-5020s")

print("\n=== Test 2: spatial diff pattern in flagged window (save heatmap) ===")
result = measure_window(3000.0, 30.0, "HEATMAP t=3000-3030s")
if result:
    mean_diff_per_pixel, _, _ = result
    # Normalize and save as an image for visual inspection
    norm = np.clip(mean_diff_per_pixel * 8, 0, 255).astype(np.uint8)  # amplify for visibility
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("scratch_frames_dahisar1/lucknow_fan_diffheatmap.jpg", heat)
    print("Saved diff heatmap to scratch_frames_dahisar1/lucknow_fan_diffheatmap.jpg")
