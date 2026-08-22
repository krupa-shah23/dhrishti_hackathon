"""
§B item 4: cheap whole-clip brightness scan to find a real mid-video lighting change
(not motion detection -- just mean grayscale level, sampled sparsely, over the full
duration of the two longest clips). A real lighting event should show as a sustained
step/ramp in mean brightness, not frame-to-frame noise.
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

CLIPS = [
    (r"data\drishti\07_seat_exchange.mkv", "DAHISAR1"),
    (r"data\drishti\06_phone_use.mp4", "LUCKNOW1"),
]

for video, cam in CLIPS:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"\n=== {cam} ({video}) duration={duration:.0f}s, total_frames={total} ===")

    # Sample 1 frame every 30s
    step_frames = int(30.0 * fps)
    samples = []
    fi = 0
    while fi < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            fi += step_frames
            continue
        gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        samples.append((fi / fps, gray_mean))
        fi += step_frames
    cap.release()

    vals = np.array([v for _, v in samples])
    print(f"  n_samples={len(samples)}  mean={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}  std={vals.std():.2f}")
    # Report per 10-minute bucket to spot drift
    print("  Per-10min bucket mean brightness:")
    bucket_sec = 600.0
    n_buckets = int(duration / bucket_sec) + 1
    for b in range(n_buckets):
        t0, t1 = b * bucket_sec, (b + 1) * bucket_sec
        bv = [v for t, v in samples if t0 <= t < t1]
        if not bv:
            continue
        print(f"    [{t0/60:5.0f}-{t1/60:5.0f}min] mean={np.mean(bv):.2f}  min={min(bv):.2f}  max={max(bv):.2f}  n={len(bv)}")

    # Flag any adjacent-sample jump > 15 gray levels (a real, fast lighting change)
    print("  Large adjacent-sample jumps (>15 gray levels):")
    found = False
    for i in range(1, len(samples)):
        t0, v0 = samples[i-1]
        t1, v1 = samples[i]
        if abs(v1 - v0) > 15:
            print(f"    t={t0:.0f}s->{t1:.0f}s : {v0:.1f} -> {v1:.1f}  (delta={v1-v0:+.1f})")
            found = True
    if not found:
        print("    none found")
