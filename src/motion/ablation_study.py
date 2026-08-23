"""
ablation_study.py

Ablation study framework for the motion detection pipeline.
Evaluates the contribution of 4 key components by disabling each individually:
  1. adaptive_baseline      - Per-seat robust Median/MAD calibration vs flat global threshold
  2. fusion                 - MOG2 + Frame-Diff OR-rule fusion vs single-method (MOG2 only)
  3. invigilator_exclusion  - Dynamic invigilator filtering/exclusion vs disabled
  4. clustering             - Spatial+temporal incident clustering vs time-only (no spatial)

Outputs results to outputs/ablation/ablation_table.csv.
"""

from __future__ import annotations
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.roi import get_rois
from src.motion.eval_metrics import evaluate_clip, compute_prf1
from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import enrich_event_with_motion_fields, segment_events, cluster_incidents


def run_pipeline_variant(
    video_path: str,
    clip_name: str = "01_phone_use.mkv",
    camera_id: str = "Camera04",
    disable_adaptive_baseline: bool = False,
    disable_fusion: bool = False,
    disable_invigilator_exclusion: bool = False,
    disable_clustering: bool = False,
    step: int = 30,
) -> List[Dict[str, Any]]:
    """
    Runs the full pipeline with specific components disabled for ablation analysis.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Safe fallback per frame_stream.py
    
    # 1. Adaptive baseline toggle is NOT CURRENTLY EVALUABLE
    # baseline.py is unused; MOG2 handles background adaptation natively.
    if disable_adaptive_baseline:
        print("[WARN] Adaptive Baseline ablation is NOT CURRENTLY EVALUABLE. baseline.py is dead code.")
    
    min_area = 300
    pipeline = P1P2TrackerPipeline(clip_name=clip_name, min_area=min_area, fps=fps)
    bridge = P2P3Bridge(missing_threshold=15, fps=fps)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Run standard pipeline frame processing
            result = pipeline.process_frame(frame, frame_idx, camera_id=camera_id)
            fused_tracks = result["fused_tracks"]

            # 2. Invigilator exclusion toggle: override invigilator_flag to False if disabled
            if disable_invigilator_exclusion:
                for ft in fused_tracks:
                    ft["invigilator_flag"] = False

            bridge.process_fused_tracks(
                fused_tracks,
                frame_index=frame_idx,
                pose_signals=result.get("pose_signals"),
                motion_intensity=result.get("motion_intensity"),
                mog2_foreground_ratio=result.get("mog2_foreground_ratio"),
            )
        frame_idx += 1

    cap.release()
    bridge.flush()
    raw_events = bridge.get_completed_events()

    enriched = [enrich_event_with_motion_fields(ev) for ev in raw_events]

    # 3. Fusion toggle: if fusion disabled, filter out events derived solely from frame-diff
    if disable_fusion:
        enriched = [ev for ev in enriched if ev.get("object_detected") or ev.get("avg_motion_intensity", 0) > 0.05]

    confirmed = segment_events(enriched, motion_threshold=0.005, min_duration_sec=0.2)

    # 4. Clustering toggle: spatial+temporal clustering vs time-only
    if disable_clustering:
        final_events = confirmed
    else:
        clusters = cluster_incidents(confirmed, time_gap_sec=5.0, camera_id=camera_id)
        final_events = [ev for cluster in clusters for ev in cluster]

    predicted = []
    for ev in final_events:
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


def run_ablation_study():
    benchmark_clips = [
        ("01_phone_use.mkv", os.path.join("data", "drishti", "01_phone_use.mkv"), "01", "Camera04"),
        ("03_mobile_usage.mkv", os.path.join("data", "drishti", "03_mobile_usage.mkv"), "03", "Camera12"),
    ]

    components = [
        ("none (baseline)", {}),
        ("adaptive_baseline", {"disable_adaptive_baseline": True}),
        ("fusion", {"disable_fusion": True}),
        ("invigilator_exclusion", {"disable_invigilator_exclusion": True}),
        ("clustering", {"disable_clustering": True}),
    ]

    ablation_results = []

    print("[+] Running baseline run with all components ENABLED...", flush=True)
    total_tp, total_fp, total_fn = 0, 0, 0
    for clip_label, video_path, gt_clip, cam_id in benchmark_clips:
        if not os.path.exists(video_path):
            continue
        preds = run_pipeline_variant(video_path, clip_name=clip_label, camera_id=cam_id)
        m = evaluate_clip(clip_name=gt_clip, pred_csv_or_json_path=preds)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]

    baseline_prf = compute_prf1(total_tp, total_fp, total_fn)
    baseline_f1 = baseline_prf["f1"]
    print(f"    Baseline Micro F1 = {baseline_f1:.4f}", flush=True)

    ablation_results.append({
        "component_disabled": "none (baseline)",
        "f1": baseline_f1,
        "f1_delta_from_baseline": 0.0,
    })

    for name, kwargs in components[1:]:
        print(f"[+] Running ablation with '{name}' DISABLED...", flush=True)
        c_tp, c_fp, c_fn = 0, 0, 0
        for clip_label, video_path, gt_clip, cam_id in benchmark_clips:
            if not os.path.exists(video_path):
                continue
            preds = run_pipeline_variant(video_path, clip_name=clip_label, camera_id=cam_id, **kwargs)
            m = evaluate_clip(clip_name=gt_clip, pred_csv_or_json_path=preds)
            c_tp += m["tp"]
            c_fp += m["fp"]
            c_fn += m["fn"]

        variant_prf = compute_prf1(c_tp, c_fp, c_fn)
        var_f1 = variant_prf["f1"]
        delta_f1 = round(var_f1 - baseline_f1, 4)

        ablation_results.append({
            "component_disabled": name,
            "f1": var_f1,
            "f1_delta_from_baseline": delta_f1,
        })
        print(f"    Disabled '{name}': F1={var_f1:.4f} (delta {delta_f1:+.4f})", flush=True)

    df_out = pd.DataFrame(ablation_results)

    out_dir = os.path.join("outputs", "ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_table.csv")
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 70, flush=True)
    print("                 PIPELINE ABLATION STUDY SUMMARY TABLE              ", flush=True)
    print("=" * 70, flush=True)
    print(df_out.to_string(index=False), flush=True)
    print("=" * 70, flush=True)
    print(f"[OK] Saved ablation study results to {out_path}\n", flush=True)


if __name__ == "__main__":
    run_ablation_study()
