"""
ablation.py -- P4/P2b: Outputs, Evaluation & Integration
Runs ablation studies by toggling individual pipeline components on/off
and measuring their F1-score contribution.

Ablation axes (from config.yaml):
  - invigilator_filter  (bool): Filter out tracks identified as invigilators
  - exam_phase_logic    (bool): Apply phase-aware risk thresholds
  - audio_fusion        (bool): Fuse audio energy into risk score

Outputs:
  outputs/ablation_table.csv
"""

import os
import csv
import copy
import time

import cv2  # type: ignore[import-not-found]
import numpy as np

from src.outputs_eval.metrics import (
    MaskMetrics,
    update_mask_metrics,
    load_events_from_csv,
    match_events,
    Event,
)


# ---------------------------------------------------------------------------
# Ablation axis definitions
# ---------------------------------------------------------------------------

ABLATION_AXES = [
    {
        "name":       "All Components (Baseline)",
        "toggle_key": None,          # nothing disabled
    },
    {
        "name":       "No Invigilator Filter",
        "toggle_key": ("risk", "invigilator_filter"),
    },
    {
        "name":       "No Exam Phase Logic",
        "toggle_key": ("risk", "exam_phase_logic"),
    },
    {
        "name":       "No Audio Fusion",
        "toggle_key": ("risk", "audio_fusion"),
    },
]


# ---------------------------------------------------------------------------
# Real ground-truth loading (ground_truth_events.csv)
# ---------------------------------------------------------------------------

def load_real_gt_events(gt_csv_path: str, video_id: str | None = None) -> list[Event]:
    """
    Loads ground_truth_events.csv (columns: event_id,video_id,start,end,
    seat_id,event_type,notes). Excludes clip5 (unresolved footage, per
    project decision). Optionally filters to a single video_id.
    """
    gt_events = []
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "")
            if vid == "clip5":
                continue
            if video_id and vid != video_id:
                continue
            gt_events.append(Event(
                start=float(row["start"]),
                end=float(row["end"]),
                label=row.get("event_type", ""),
            ))
    return gt_events


# ---------------------------------------------------------------------------
# Mock ablation runner (uses synthetic events until real pipeline lands)
# ---------------------------------------------------------------------------

def _mock_run_pipeline_for_ablation(
    video_path: str,
    config: dict,
) -> tuple[list[Event], float]:
    """
    Placeholder that simulates processing the video and returns
    predicted events + elapsed time.

    TODO: replace with a call to the real pipeline (P1 motion + P2a
    detector via the tracker bridge) once the smoke test (§3, joint
    execution doc) passes. Swap this function's body only — callers
    and the CSV output shape stay the same.

    The mock applies a simple heuristic: disabling components
    reduces the number of detected events (simulating real F1 impact).
    """
    invigilator_filter = config.get("risk", {}).get("invigilator_filter", True)
    exam_phase_logic   = config.get("risk", {}).get("exam_phase_logic",   True)
    audio_fusion       = config.get("risk", {}).get("audio_fusion",       True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_s = total_frames / fps

    base_event_count = int(duration_s / 3.0)

    if not invigilator_filter:
        base_event_count = int(base_event_count * 1.30)
    if not exam_phase_logic:
        base_event_count = int(base_event_count * 1.20)
    if not audio_fusion:
        base_event_count = int(base_event_count * 1.10)

    events = []
    step = duration_s / base_event_count if base_event_count > 0 else duration_s
    for i in range(base_event_count):
        t_start = i * step
        t_end   = t_start + 2.0
        events.append(Event(start=t_start, end=min(t_end, duration_s)))

    elapsed = 0.5   # mocked timing
    return events, elapsed


def _build_mock_gt_events(video_path: str) -> list[Event]:
    """Fallback synthetic GT (one event every ~5s) — used only when no
    real ground_truth_events.csv path is supplied."""
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_s = total_frames / fps
    gt_events  = []

    t = 4.0
    while t < duration_s:
        gt_events.append(Event(start=t, end=t + 2.5))
        t += 5.0

    return gt_events


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    video_path: str,
    base_config: dict,
    gt_events: list[Event] | None = None,
    gt_events_csv: str | None = None,
    video_id: str | None = None,
    overlap_threshold: float = 0.50,
) -> list[dict]:
    """
    Runs the full ablation study for all axes and returns results.

    GT precedence: explicit gt_events list > gt_events_csv (real,
    ground_truth_events.csv) > mock GT (last resort, flag in report).

    Args:
        video_path        (str):         Path to input video.
        base_config       (dict):        Full config.yaml loaded as dict.
        gt_events         (list[Event]): Pre-loaded GT, optional override.
        gt_events_csv     (str):         Path to ground_truth_events.csv.
        video_id          (str):         video_id to filter GT rows by.
        overlap_threshold (float):       Temporal IoU threshold for matching.

    Returns:
        list[dict]: One result dict per ablation row.
    """
    using_mock_gt = False
    if gt_events is None:
        if gt_events_csv:
            gt_events = load_real_gt_events(gt_events_csv, video_id)
        else:
            gt_events = _build_mock_gt_events(video_path)
            using_mock_gt = True

    gt_label = "MOCK" if using_mock_gt else "REAL"
    print(f"[ablation] GT events loaded: {len(gt_events)} ({gt_label})")
    print(f"[ablation] Running {len(ABLATION_AXES)} ablation configurations...\n")

    rows = []

    for axis in ABLATION_AXES:
        cfg = copy.deepcopy(base_config)

        toggle = axis["toggle_key"]
        if toggle is not None:
            section, key = toggle
            if section in cfg:
                cfg[section][key] = False

        t_start = time.time()
        pred_events, _ = _mock_run_pipeline_for_ablation(video_path, cfg)
        elapsed = time.time() - t_start

        event_metrics = match_events(pred_events, gt_events, overlap_threshold)

        row = {
            "configuration":      axis["name"],
            "component_disabled": toggle[1] if toggle else "none",
            "gt_source":          gt_label,
            "pred_events":        len(pred_events),
            "gt_events":          len(gt_events),
            "tp":                 event_metrics.tp_events,
            "fp":                 event_metrics.fp_events,
            "fn":                 event_metrics.fn_events,
            "precision":          round(event_metrics.precision, 4),
            "recall":             round(event_metrics.recall,    4),
            "f1":                 round(event_metrics.f1,        4),
            "elapsed_s":          round(elapsed, 2),
        }
        rows.append(row)

        print(f"  [{axis['name']:<35}]  "
              f"F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")

    baseline_f1 = rows[0]["f1"]
    print(f"\n[ablation] F1 deltas vs. baseline (F1={baseline_f1:.4f}):")
    for r in rows[1:]:
        delta = r["f1"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        print(f"  Removing {r['component_disabled']:<25} -> F1 delta = {sign}{delta:.4f}")

    if using_mock_gt:
        print("\n[ablation] WARNING: results use MOCK ground truth — "
              "do not present these as final numbers. Pass --gt-events "
              "to use ground_truth_events.csv.")

    return rows


# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------

def save_ablation_csv(rows: list[dict], output_path: str) -> None:
    """Saves ablation results to outputs/ablation_table.csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[ablation] Table saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="P4/P2b Ablation Study Runner")
    parser.add_argument("--video",      required=True, help="Path to input video")
    parser.add_argument("--video-id",   default=None,  help="video_id matching ground_truth_events.csv")
    parser.add_argument("--gt-events",  default=None,  help="Path to ground_truth_events.csv")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--out",        default="outputs/ablation_table.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    rows = run_ablation(
        args.video,
        base_config,
        gt_events_csv=args.gt_events,
        video_id=args.video_id,
    )
    save_ablation_csv(rows, args.out)