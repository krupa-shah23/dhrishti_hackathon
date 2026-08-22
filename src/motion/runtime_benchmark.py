"""
runtime_benchmark.py

Times full_pipeline on the longest clip currently present in data/ (checked against manifest.csv).
Computes:
  - minutes of footage per minute of compute
  - peak memory usage (via psutil if available, catching ImportError if unavailable)

Outputs:
  - outputs/runtime/runtime_report.json
"""

import os
import sys
import time
import json
import cv2
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.motion.benchmark_stages import full_pipeline

def find_longest_clip():
    """
    Finds the longest available video clip listed in manifest.csv or found on disk.
    """
    manifest_path = "manifest.csv"
    longest_clip = None
    max_duration = -1.0

    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            clip_name = str(row["filename"])
            duration = float(row["duration_sec"])
            candidates = [
                row.get("path"),
                os.path.join("data", "drishti", clip_name),
                os.path.join("data", "drishti", clip_name),
                os.path.join("data", clip_name),
            ]
            found_path = None
            for c in candidates:
                if c and os.path.exists(c):
                    found_path = c
                    break

            if found_path and duration > max_duration:
                max_duration = duration
                longest_clip = {
                    "video_id": row.get("video_id"),
                    "filename": clip_name,
                    "duration_sec": duration,
                    "path": found_path
                }

    if not longest_clip:
        # Fallback search if manifest clips are not found at direct paths
        search_dirs = [os.path.join("data", "drishti"), "data"]
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for f in os.listdir(s_dir):
                    if f.endswith(('.mkv', '.mp4', '.avi')):
                        p = os.path.join(s_dir, f)
                        cap = cv2.VideoCapture(p)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                            dur = frames / fps if fps > 0 else 0
                            cap.release()
                            if dur > max_duration:
                                max_duration = dur
                                longest_clip = {
                                    "video_id": f,
                                    "filename": f,
                                    "duration_sec": dur,
                                    "path": p
                                }

    return longest_clip

def run_runtime_benchmark():
    output_dir = os.path.join("outputs", "runtime")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "runtime_report.json")

    clip_info = find_longest_clip()
    if not clip_info:
        print("[ERROR] No valid video clips found on disk!")
        return

    video_path = clip_info["path"]
    filename = clip_info["filename"]
    duration_sec = clip_info["duration_sec"]
    minutes_of_footage = duration_sec / 60.0

    print(f"[+] Running runtime benchmark on longest available clip: {filename} ({duration_sec:.2f}s / {minutes_of_footage:.2f} min)...", flush=True)

    # Safely try importing psutil for peak memory
    psutil_available = False
    process = None
    try:
        import psutil
        psutil_available = True
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        psutil_available = False
        print("[INFO] psutil not installed")
        mem_before = None

    t0 = time.time()
    preds = full_pipeline(video_path, clip_name=filename)
    compute_runtime_sec = time.time() - t0
    minutes_of_compute = compute_runtime_sec / 60.0

    if psutil_available and process:
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_memory = f"{round(max(mem_before, mem_after), 2)} MB"
    else:
        peak_memory = "psutil not installed"

    footage_per_compute = minutes_of_footage / minutes_of_compute if minutes_of_compute > 0 else 0.0

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "longest_clip": filename,
        "clip_path": video_path,
        "clip_duration_sec": round(duration_sec, 2),
        "minutes_of_footage": round(minutes_of_footage, 4),
        "compute_runtime_sec": round(compute_runtime_sec, 4),
        "minutes_of_compute": round(minutes_of_compute, 4),
        "minutes_of_footage_per_minute_of_compute": round(footage_per_compute, 2),
        "total_predicted_events": len(preds),
        "peak_memory": peak_memory,
        "psutil_installed": psutil_available
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("                RUNTIME BENCHMARK REPORT                ")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)
    print(f"[OK] Saved runtime report to {report_file}\n")

if __name__ == "__main__":
    run_runtime_benchmark()
