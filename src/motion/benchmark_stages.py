"""
benchmark_stages.py

Benchmarks 4 motion detection pipeline variants across short and long video clips:
  1. frame_diff_only  - Pure consecutive frame differencing thresholding
  2. mog2_only        - MOG2 background subtractor foreground mask
  3. mog2_plus_roi    - MOG2 foreground mask + ROI seat region extraction
  4. full_pipeline    - Integrated pipeline (MOG2 + Frame-Diff fusion, ROI, Tracking, P2P3Bridge, Clustering)

Outputs results to outputs/benchmarks/stage_benchmark.csv.
"""

from __future__ import annotations
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.roi import get_rois
from src.motion.eval_metrics import evaluate_clip
from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import enrich_event_with_motion_fields, segment_events, cluster_incidents


# --- Variant 1: Frame-Diff Only ---
def frame_diff_only(video_path: str, diff_threshold: int = 25, min_event_frames: int = 2, step: int = 30) -> List[Dict[str, Any]]:
    """
    Pipeline Variant 1: Pure frame differencing.
    Computes absdiff(gray_t, gray_{t-1}) > diff_threshold and groups active motion frames into temporal events.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_gray = None
    frame_idx = 0
    active_start_frame = None
    events = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_ratio = float(np.count_nonzero(diff > diff_threshold)) / diff.size
            
            if motion_ratio > 0.01:
                if active_start_frame is None:
                    active_start_frame = frame_idx
            else:
                if active_start_frame is not None:
                    dur_frames = frame_idx - active_start_frame
                    if dur_frames >= min_event_frames:
                        events.append({
                            "event_id": f"fd_{len(events)}",
                            "start_time": active_start_frame / fps,
                            "end_time": frame_idx / fps,
                            "start_sec": active_start_frame / fps,
                            "end_sec": frame_idx / fps,
                            "event_type": "motion_event",
                        })
                    active_start_frame = None
        prev_gray = gray
        frame_idx += step

    if active_start_frame is not None and (frame_idx - active_start_frame) >= min_event_frames:
        events.append({
            "event_id": f"fd_{len(events)}",
            "start_time": active_start_frame / fps,
            "end_time": frame_idx / fps,
            "start_sec": active_start_frame / fps,
            "end_sec": frame_idx / fps,
            "event_type": "motion_event",
        })

    cap.release()
    return events


# --- Variant 2: MOG2 Only ---
def mog2_only(video_path: str, min_event_frames: int = 2, step: int = 30) -> List[Dict[str, Any]]:
    """
    Pipeline Variant 2: MOG2 background subtraction only.
    Uses MotionEstimator MOG2 binary mask without fusion or tracking.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    estimator = MotionEstimator()
    frame_idx = 0
    active_start_frame = None
    events = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        _, mog2_mask = estimator.get_motion_mask(frame)
        motion_ratio = float(np.count_nonzero(mog2_mask)) / mog2_mask.size

        if motion_ratio > 0.008:
            if active_start_frame is None:
                active_start_frame = frame_idx
        else:
            if active_start_frame is not None:
                dur_frames = frame_idx - active_start_frame
                if dur_frames >= min_event_frames:
                    events.append({
                        "event_id": f"mog2_{len(events)}",
                        "start_time": active_start_frame / fps,
                        "end_time": frame_idx / fps,
                        "start_sec": active_start_frame / fps,
                        "end_sec": frame_idx / fps,
                        "event_type": "motion_event",
                    })
                active_start_frame = None
        frame_idx += step

    if active_start_frame is not None and (frame_idx - active_start_frame) >= min_event_frames:
        events.append({
            "event_id": f"mog2_{len(events)}",
            "start_time": active_start_frame / fps,
            "end_time": frame_idx / fps,
            "start_sec": active_start_frame / fps,
            "end_sec": frame_idx / fps,
            "event_type": "motion_event",
        })

    cap.release()
    return events


# --- Variant 3: MOG2 + ROI ---
def mog2_plus_roi(video_path: str, camera_id: str = "Camera12", step: int = 30) -> List[Dict[str, Any]]:
    """
    Pipeline Variant 3: MOG2 background subtraction + seat-based ROI region extraction.
    Groups per-seat ROI bounding boxes into seat-tagged events.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    estimator = MotionEstimator()
    frame_idx = 0
    seat_active_start: Dict[str, int] = {}
    events = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        _, mog2_mask = estimator.get_motion_mask(frame)
        rois = get_rois(mog2_mask, min_area=300, camera_id=camera_id)
        current_seats = set()

        for roi in rois:
            seat_id = roi.get("seat_id", "unknown") if isinstance(roi, dict) else "unknown"
            if seat_id != "unknown":
                current_seats.add(seat_id)
                if seat_id not in seat_active_start:
                    seat_active_start[seat_id] = frame_idx

        for seat_id in list(seat_active_start.keys()):
            if seat_id not in current_seats:
                start_f = seat_active_start.pop(seat_id)
                if (frame_idx - start_f) >= 3:
                    events.append({
                        "event_id": f"roi_{len(events)}",
                        "seat_id": seat_id,
                        "start_time": start_f / fps,
                        "end_time": frame_idx / fps,
                        "start_sec": start_f / fps,
                        "end_sec": frame_idx / fps,
                        "event_type": "seat_motion",
                    })
        frame_idx += step

    for seat_id, start_f in seat_active_start.items():
        if (frame_idx - start_f) >= 3:
            events.append({
                "event_id": f"roi_{len(events)}",
                "seat_id": seat_id,
                "start_time": start_f / fps,
                "end_time": frame_idx / fps,
                "start_sec": start_f / fps,
                "end_sec": frame_idx / fps,
                "event_type": "seat_motion",
            })

    cap.release()
    return events


# --- Variant 4: Full Pipeline ---
def full_pipeline(video_path: str, clip_name: str = "01_phone_use.mkv", step: int = 30) -> List[Dict[str, Any]]:
    """
    Pipeline Variant 4: Complete end-to-end pipeline.
    Combines MOG2+Frame-Diff fusion, ROI, Tracking, P2P3Bridge, event enrichment, hysteresis, and incident clustering.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Safe fallback per frame_stream.py
    pipeline = P1P2TrackerPipeline(clip_name=clip_name, min_area=300, fps=fps)
    bridge = P2P3Bridge(missing_threshold=15, fps=fps)

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, frame_idx)
        bridge.process_fused_tracks(
            result["fused_tracks"],
            frame_index=frame_idx,
            pose_signals=result.get("pose_signals"),
            motion_intensity=result.get("motion_intensity"),
            mog2_foreground_ratio=result.get("mog2_foreground_ratio"),
        )
        frame_idx += step

    cap.release()
    bridge.flush()
    raw_events = bridge.get_completed_events()

    enriched = [enrich_event_with_motion_fields(ev) for ev in raw_events]
    confirmed = segment_events(enriched, motion_threshold=0.005, min_duration_sec=0.2)
    clusters = cluster_incidents(confirmed, time_gap_sec=5.0)

    predicted = []
    for c in clusters:
        for ev in c:
            predicted.append({
                "event_id": ev.get("event_id"),
                "seat_id": ev.get("seat_ids", ["unknown"])[0] if ev.get("seat_ids") else "unknown",
                "start_time": ev.get("start_time", 0.0),
                "end_time": ev.get("end_time", 0.0),
                "start_sec": ev.get("start_time", 0.0),
                "end_sec": ev.get("end_time", 0.0),
                "event_type": ev.get("event_type", "motion_event"),
            })
    return predicted


def run_benchmark():
    benchmarks = [
        ("01_phone_use.mkv", os.path.join("data", "drishti", "01_phone_use.mkv"), "01"),
        ("03_mobile_usage.mkv", os.path.join("data", "drishti", "03_mobile_usage.mkv"), "03"),
    ]

    variants = [
        ("frame_diff_only", frame_diff_only),
        ("mog2_only", mog2_only),
        ("mog2_plus_roi", mog2_plus_roi),
        ("full_pipeline", full_pipeline),
    ]

    results = []

    for var_name, var_fn in variants:
        for clip_label, video_path, gt_clip in benchmarks:
            if not os.path.exists(video_path):
                print(f"[WARN] Video not found: {video_path}", flush=True)
                continue

            print(f"[+] Benchmarking {var_name} on {clip_label}...", flush=True)
            t0 = time.time()
            if var_name == "full_pipeline":
                preds = var_fn(video_path, clip_name=clip_label)
            else:
                preds = var_fn(video_path)
            runtime_sec = time.time() - t0

            eval_metrics = evaluate_clip(clip_name=gt_clip, pred_csv_or_json_path=preds)
            results.append({
                "variant": var_name,
                "clip": clip_label,
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "runtime_sec": round(runtime_sec, 3),
            })
            print(f"    Done {var_name} on {clip_label}: F1={eval_metrics['f1']:.4f}, time={runtime_sec:.2f}s", flush=True)

    df_out = pd.DataFrame(results)

    out_dir = os.path.join("outputs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage_benchmark.csv")
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 70, flush=True)
    print("                 PIPELINE STAGE BENCHMARK SUMMARY TABLE             ", flush=True)
    print("=" * 70, flush=True)
    print(df_out.to_string(index=False), flush=True)
    print("=" * 70, flush=True)
    print(f"[OK] Saved stage benchmark results to {out_path}\n", flush=True)


if __name__ == "__main__":
    run_benchmark()
