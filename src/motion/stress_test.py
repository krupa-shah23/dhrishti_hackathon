"""
stress_test.py

Stress tests the motion detection pipeline by generating degraded video clips
(Gaussian noise, reduced brightness, occlusion box) and evaluating F1 degradation
relative to the clean baseline.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.motion.benchmark_stages import full_pipeline
from src.motion.eval_metrics import evaluate_clip

def add_gaussian_noise(frame: np.ndarray) -> np.ndarray:
    """Adds zero-mean Gaussian noise to a video frame."""
    noise = np.random.normal(0, 25, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def reduce_brightness(frame: np.ndarray) -> np.ndarray:
    """Reduces the brightness of a video frame."""
    return cv2.convertScaleAbs(frame, alpha=0.3, beta=0)

def add_occlusion_box(frame: np.ndarray) -> np.ndarray:
    """Adds a black occlusion rectangle in the center of the frame."""
    degraded = frame.copy()
    h, w = frame.shape[:2]
    h_start, h_end = int(h * 0.35), int(h * 0.65)
    w_start, w_end = int(w * 0.35), int(w * 0.65)
    degraded[h_start:h_end, w_start:w_end] = 0
    return degraded

def generate_degraded_video(input_video_path: str, output_video_path: str, degradation_fn) -> None:
    """
    Reads input video frame by frame, applies degradation_fn, and writes to output_video_path.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for degradation: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    np.random.seed(42)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        degraded_frame = degradation_fn(frame)
        out.write(degraded_frame)

    cap.release()
    out.release()

def run_stress_test():
    output_dir = os.path.join("outputs", "stress_test")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    skipped_file = os.path.join(output_dir, "skipped_clips.txt")
    results_file = os.path.join(output_dir, "stress_results.csv")

    # Step 1: Log missing clips in data/drishti/
    drishti_dir = os.path.join("data", "drishti")
    skipped_items = []
    expected_drishti_clips = [
        "01_phone_use.mkv", "02_phone_use.mkv", "03_mobile_usage.mkv",
        "04_candidate_talking.mkv", "05_crowd_reception.mp4", "06_phone_use.mp4",
        "07_seat_exchange.mkv", "08_seat12_copying.mkv"
    ]
    if not os.path.exists(drishti_dir):
        skipped_items.append(f"{drishti_dir}/ (Directory missing)")
        for clip in expected_drishti_clips:
            skipped_items.append(f"{clip} ({os.path.join(drishti_dir, clip)} - Clip missing)")
    else:
        for clip in expected_drishti_clips:
            clip_path = os.path.join(drishti_dir, clip)
            if not os.path.exists(clip_path):
                skipped_items.append(f"{clip} ({clip_path} - Clip missing)")

    with open(skipped_file, "w", encoding="utf-8") as f:
        for item in skipped_items:
            f.write(f"{item}\n")
    print(f"[INFO] Logged skipped clips to {skipped_file}")

    # Step 2: Available clean clip (01_phone_use.mkv)
    clean_clip_path = os.path.join("data", "clg_dataset", "01_phone_use.mkv")
    if not os.path.exists(clean_clip_path):
        print(f"[ERROR] Clean clip not found at {clean_clip_path}")
        return

    gt_path = os.path.join("src", "motion", "ground_truth_01.csv")

    degradations = [
        ("original", None),
        ("gaussian_noise", add_gaussian_noise),
        ("reduced_brightness", reduce_brightness),
        ("occlusion_box", add_occlusion_box),
    ]

    results = []
    clean_f1 = None

    for cond_name, deg_fn in degradations:
        print(f"[+] Running condition: {cond_name}...", flush=True)
        if cond_name == "original":
            video_to_run = clean_clip_path
        else:
            video_to_run = os.path.join(tmp_dir, f"{cond_name}.mp4")
            generate_degraded_video(clean_clip_path, video_to_run, deg_fn)

        preds = full_pipeline(video_to_run, clip_name="01_phone_use.mkv", step=1)
        metrics = evaluate_clip(clip_name="01", pred_csv_or_json_path=preds, gt_csv_path=gt_path)
        f1 = float(metrics["f1"])

        if cond_name == "original":
            clean_f1 = f1
            deg_pct = 0.0
        else:
            if clean_f1 is not None and clean_f1 > 0:
                deg_pct = round(((clean_f1 - f1) / clean_f1) * 100.0, 2)
            else:
                deg_pct = 0.0

        results.append({
            "condition": cond_name,
            "f1": round(f1, 4),
            "f1_degradation_pct_vs_clean": deg_pct
        })
        print(f"    Finished {cond_name}: F1={f1:.4f}, Degradation={deg_pct}%", flush=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(results_file, index=False)
    print("\n" + "=" * 60)
    print("                STRESS TEST RESULTS                ")
    print("=" * 60)
    print(df_out.to_string(index=False))
    print("=" * 60)
    print(f"[OK] Saved stress test results to {results_file}\n")

if __name__ == "__main__":
    run_stress_test()
