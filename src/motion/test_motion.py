"""
test_motion.py
Owner: P1 - Motion & ROI

Quick sanity checks:
- On a mostly-static clip (backdoor/cubicle), mask should be near-blank
  in empty frames.
- On a clip with movement, mask should highlight only the moving subject.
- Saves a few sample mask frames + an overlay so you can eyeball results.

Update CLIP_PATHS below to point at your actual clip folders.
"""

import os
import cv2
import csv
try:
    from .motion import MotionEstimator, motion_intensity
except ImportError:
    from motion import MotionEstimator, motion_intensity


print("test_motion.py started")

# ---- Update these paths to match your /data structure ----
CLIP_PATHS = {
    "cubicle": "data/shadow/cubicle/input",
    "backdoor": "data/shadow/backdoor/input",
    "copyMachine": "data/shadow/copyMachine/input",
    "tramCrossroad_1fps": "data/lowFramerate/tramCrossroad_1fps/input",
    "fountain02": "data/dynamicBackground/fountain02/input",
}

OUTPUT_DIR = "outputs/masks"
SAMPLE_EVERY_N_FRAMES = 20  # save 1 out of every N frames as a sanity check


def load_frames_from_folder(folder_path):
    """
    CDNet2014 clips ship as numbered image files (in000001.jpg etc.)
    rather than a single video file. This loads them in order.
    """
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".bmp")))
    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            yield fname, frame


def run_check(clip_name, folder_path):
    print(f"\n--- Running motion check on: {clip_name} ---")
    if not os.path.isdir(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}")
        return

    out_dir = os.path.join(OUTPUT_DIR, clip_name)
    os.makedirs(out_dir, exist_ok=True)

    estimator = MotionEstimator()
    intensities = []

    for i, (fname, frame) in enumerate(load_frames_from_folder(folder_path)):
        mask = estimator.get_motion_mask(frame)
        score = motion_intensity(mask)
        intensities.append(score)

        if i % SAMPLE_EVERY_N_FRAMES == 0:
            # Save mask + overlay side-by-side so you can visually check
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(frame, 0.7, mask_bgr, 0.5, 0)
            combined = cv2.hconcat([frame, mask_bgr, overlay])
            cv2.imwrite(os.path.join(out_dir, f"sample_{i:05d}.jpg"), combined)

    if intensities:
        avg = sum(intensities) / len(intensities)
        peak = max(intensities)
        print(f"  Frames processed: {len(intensities)}")
        print(f"  Avg motion intensity: {avg:.4f}")
        print(f"  Peak motion intensity: {peak:.4f}")
        print(f"  Sample frames saved to: {out_dir}")

        log_dir = "outputs/benchmark_logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"{clip_name}_motion_intensity.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "motion_intensity"])
            for idx, score in enumerate(intensities):
                writer.writerow([idx, score])
        print(f"  Intensity log saved to: {csv_path}")
    else:
        print("  [WARN] No frames were processed — check folder contents.")


if __name__ == "__main__":
    for name, path in CLIP_PATHS.items():
        run_check(name, path)