"""
ablation.py -- P4: Outputs, Evaluation & Integration
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

# Each entry describes one ablation: a human label, and which config key
# it disables (set to False). The baseline has all components enabled.
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
# Mock ablation runner (uses synthetic events for demonstration)
# ---------------------------------------------------------------------------

def _mock_run_pipeline_for_ablation(
    video_path: str,
    config: dict,
) -> tuple[list[Event], float]:
    """
    Placeholder that simulates processing the video and returns
    predicted events + elapsed time.

    When P1/P2/P3 real modules are ready, replace this with a call
    to main.run_pipeline() that returns the events list.

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

    # Simulate: each active component prunes false positives
    # Baseline: detect events every ~3 seconds
    base_event_count = int(duration_s / 3.0)

    if not invigilator_filter:
        # Without invigilator filter, ~30% more false events appear
        base_event_count = int(base_event_count * 1.30)
    if not exam_phase_logic:
        # Without phase logic, ~20% more false events during quiet periods
        base_event_count = int(base_event_count * 1.20)
    if not audio_fusion:
        # Without audio, ~10% more false positives (pure visual noise)
        base_event_count = int(base_event_count * 1.10)

    # Build synthetic predicted events
    events = []
    step = duration_s / base_event_count if base_event_count > 0 else duration_s
    for i in range(base_event_count):
        t_start = i * step
        t_end   = t_start + 2.0   # 2-second event duration
        events.append(Event(start=t_start, end=min(t_end, duration_s)))

    elapsed = 0.5   # mocked timing
    return events, elapsed


def _build_mock_gt_events(video_path: str) -> list[Event]:
    """
    Generates mock ground-truth events (one suspicious event every ~5 seconds).
    Replace with real annotation CSV loading when demo_clips ground truth is ready.
    """
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
    overlap_threshold: float = 0.50,
) -> list[dict]:
    """
    Runs the full ablation study for all axes and returns results.

    Args:
        video_path        (str):         Path to input video.
        base_config       (dict):        Full config.yaml loaded as dict.
        gt_events         (list[Event]): Ground-truth event annotations.
                                         If None, uses mock GT.
        overlap_threshold (float):       Temporal IoU threshold for event matching.

    Returns:
        list[dict]: One result dict per ablation row.
    """
    if gt_events is None:
        gt_events = _build_mock_gt_events(video_path)

    print(f"[ablation] GT events loaded: {len(gt_events)}")
    print(f"[ablation] Running {len(ABLATION_AXES)} ablation configurations...\n")

    rows = []

    for axis in ABLATION_AXES:
        # Deep copy config so we don't mutate shared state
        cfg = copy.deepcopy(base_config)

        # Toggle the ablated component off
        toggle = axis["toggle_key"]
        if toggle is not None:
            section, key = toggle
            if section in cfg:
                cfg[section][key] = False

        # Run pipeline (mock or real)
        t_start = time.time()
        pred_events, _ = _mock_run_pipeline_for_ablation(video_path, cfg)
        elapsed = time.time() - t_start

        # Evaluate
        event_metrics = match_events(pred_events, gt_events, overlap_threshold)

        row = {
            "configuration":      axis["name"],
            "component_disabled": toggle[1] if toggle else "none",
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

    # Print F1 delta vs. baseline
    baseline_f1 = rows[0]["f1"]
    print(f"\n[ablation] F1 deltas vs. baseline (F1={baseline_f1:.4f}):")
    for r in rows[1:]:
        delta = r["f1"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        print(f"  Removing {r['component_disabled']:<25} -> F1 delta = {sign}{delta:.4f}")

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

    parser = argparse.ArgumentParser(description="P4 Ablation Study Runner")
    parser.add_argument("--video",  required=True, help="Path to input video")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out",    default="outputs/ablation_table.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    rows = run_ablation(args.video, base_config)
    save_ablation_csv(rows, args.out)
