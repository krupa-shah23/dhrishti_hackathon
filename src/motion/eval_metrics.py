"""
eval_metrics.py

Evaluation metrics module for motion/ROI and temporal event detection pipelines.
Provides interval IoU overlap ratio calculation, event matching with seat_id validation,
Precision/Recall/F1 calculation (micro-averaged across clips), and evaluation reporting.

Per Master Doc §8: Raw accuracy is explicitly excluded as class imbalance makes it misleading.
"""

from __future__ import annotations
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

# Mapping of clip identifiers and video filenames to ground truth CSV paths (1-to-1)
GT_FILE_MAPPING = {
    "01": "src/motion/ground_truth_01.csv",
    "01_phone_use": "src/motion/ground_truth_01.csv",
    "01_phone_use.mkv": "src/motion/ground_truth_01.csv",
    "02": "src/motion/ground_truth_02.csv",
    "02_phone_use": "src/motion/ground_truth_02.csv",
    "02_phone_use.mkv": "src/motion/ground_truth_02.csv",
    "03": "src/motion/ground_truth_03.csv",
    "03_mobile_usage": "src/motion/ground_truth_03.csv",
    "03_mobile_usage.mkv": "src/motion/ground_truth_03.csv",
    "04": "src/motion/ground_truth_04.csv",
    "04_candidate_talking": "src/motion/ground_truth_04.csv",
    "04_candidate_talking.mkv": "src/motion/ground_truth_04.csv",
    "05": "src/motion/ground_truth_05.csv",
    "05_crowd_reception": "src/motion/ground_truth_05.csv",
    "05_crowd_reception.mkv": "src/motion/ground_truth_05.csv",
    "06": "src/motion/ground_truth_06.csv",
    "06_phone_use": "src/motion/ground_truth_06.csv",
    "06_phone_use.mp4": "src/motion/ground_truth_06.csv",
    "07": "src/motion/ground_truth_07.csv",
    "07_seat_exchange": "src/motion/ground_truth_07.csv",
    "07_seat_exchange.mkv": "src/motion/ground_truth_07.csv",
    "08": "src/motion/ground_truth_08.csv",
    "08_seat12_copying": "src/motion/ground_truth_08.csv",
    "08_seat12_copying.mkv": "src/motion/ground_truth_08.csv",
}


def _extract_start_end(event: Union[Dict[str, Any], Any]) -> Tuple[float, float]:
    """Extract start and end timestamps from an event dict/object."""
    if isinstance(event, dict):
        start = event.get("start_sec", event.get("start_time", event.get("start", 0.0)))
        end = event.get("end_sec", event.get("end_time", event.get("end", 0.0)))
        # Check nested timestamps list if present e.g. [{"start": ..., "end": ...}]
        if "timestamps" in event and isinstance(event["timestamps"], list) and len(event["timestamps"]) > 0:
            ts = event["timestamps"][0]
            if isinstance(ts, dict):
                start = ts.get("start", start)
                end = ts.get("end", end)
    else:
        start = getattr(event, "start_sec", getattr(event, "start_time", getattr(event, "start", 0.0)))
        end = getattr(event, "end_sec", getattr(event, "end_time", getattr(event, "end", 0.0)))
    return float(start), float(end)


def time_overlap_ratio(pred_event: Union[Dict[str, Any], Any], gt_event: Union[Dict[str, Any], Any]) -> float:
    """
    Computes IoU-style temporal overlap ratio between two [start, end] intervals
    as overlap_duration / union_duration.

    For point-events (duration 0), returns 1.0 if the point falls within the
    other interval, otherwise 0.0.
    """
    pred_start, pred_end = _extract_start_end(pred_event)
    gt_start, gt_end = _extract_start_end(gt_event)

    pred_dur = max(0.0, pred_end - pred_start)
    gt_dur = max(0.0, gt_end - gt_start)

    # Handle point-events natively
    if gt_dur == 0.0:
        return 1.0 if pred_start <= gt_start <= pred_end else 0.0
    if pred_dur == 0.0:
        return 1.0 if gt_start <= pred_start <= gt_end else 0.0

    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    overlap = max(0.0, intersection_end - intersection_start)

    union = pred_dur + gt_dur - overlap

    if union <= 0.0:
        return 0.0
    return float(overlap / union)


def _extract_seat_ids(event: Union[Dict[str, Any], Any]) -> set:
    """Extract seat IDs from an event if present as a set."""
    seats = set()
    if isinstance(event, dict):
        if "seat_ids" in event and isinstance(event["seat_ids"], list):
            seats.update([str(s).strip() for s in event["seat_ids"] if s])
        elif "seat_id" in event or "seat" in event:
            s = event.get("seat_id", event.get("seat"))
            if s is not None and str(s).strip() != "" and str(s).strip().lower() != "nan":
                seats.add(str(s).strip())
    else:
        if hasattr(event, "seat_ids") and isinstance(event.seat_ids, list):
            seats.update([str(s).strip() for s in event.seat_ids if s])
        elif hasattr(event, "seat_id") or hasattr(event, "seat"):
            s = getattr(event, "seat_id", getattr(event, "seat", None))
            if s is not None and str(s).strip() != "" and str(s).strip().lower() != "nan":
                seats.add(str(s).strip())
    return seats


def match_events(
    pred_events: List[Union[Dict[str, Any], Any]],
    gt_events: List[Union[Dict[str, Any], Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    For each ground-truth event, finds the best-matching predicted event by time_overlap_ratio.
    A match counts as True Positive (TP) only if overlap_ratio >= threshold AND seat_id matches
    (if seat_id is present and non-empty in both events).

    Unmatched GT events count as False Negative (FN).
    Unmatched predicted events count as False Positive (FP).

    Returns dict:
        {
            "tp": int,
            "fp": int,
            "fn": int,
            "matches": list of {"pred": pred_event, "gt": gt_event, "overlap_ratio": float}
        }
    """
    if not gt_events and not pred_events:
        return {"tp": 0, "fp": 0, "fn": 0, "matches": []}

    matched_pred_indices = set()
    matches = []

    # Sort GT events by start time for consistent evaluation order
    sorted_gt = sorted(enumerate(gt_events), key=lambda x: _extract_start_end(x[1])[0])

    for gt_idx, gt_ev in sorted_gt:
        gt_seats = _extract_seat_ids(gt_ev)
        best_pred_idx = None
        best_overlap = 0.0

        for p_idx, pred_ev in enumerate(pred_events):
            if p_idx in matched_pred_indices:
                continue

            # Validate seat_id compatibility if present in both
            pred_seats = _extract_seat_ids(pred_ev)
            if gt_seats and pred_seats and not (gt_seats & pred_seats):
                continue

            overlap = time_overlap_ratio(pred_ev, gt_ev)
            if overlap >= threshold and overlap > best_overlap:
                best_overlap = overlap
                best_pred_idx = p_idx

        if best_pred_idx is not None:
            matched_pred_indices.add(best_pred_idx)
            matches.append({
                "pred": pred_events[best_pred_idx],
                "gt": gt_ev,
                "overlap_ratio": best_overlap,
            })

    tp = len(matches)
    fn = len(gt_events) - tp
    fp = len(pred_events) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
    }


def compute_prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 score from confusion counts.
    Safely handles divide-by-zero by returning 0.0 (never NaN or crashing).

    Per Master Doc §8: Raw accuracy is explicitly excluded due to class imbalance.
    """
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _load_events_from_path(path: str) -> List[Dict[str, Any]]:
    """Loads event list from a CSV or JSON file path."""
    if not path or not os.path.exists(path):
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "events" in data:
                return data["events"]
            return [data]
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    return []


def evaluate_clip(
    clip_name: str,
    pred_csv_or_json_path: Optional[Union[str, List[Dict[str, Any]]]] = None,
    gt_csv_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth for a single clip.

    Loads GT and predictions, runs match_events and compute_prf1,
    and returns a per-clip metrics dictionary.
    """
    # Resolve GT path
    if gt_csv_path is None:
        gt_csv_path = GT_FILE_MAPPING.get(clip_name, f"src/motion/ground_truth_{clip_name}.csv")

    gt_events = _load_events_from_path(gt_csv_path)

    # Resolve predicted events
    if isinstance(pred_csv_or_json_path, list):
        pred_events = pred_csv_or_json_path
    elif isinstance(pred_csv_or_json_path, str):
        pred_events = _load_events_from_path(pred_csv_or_json_path)
    else:
        pred_events = []

    match_result = match_events(pred_events, gt_events, threshold=threshold)
    tp, fp, fn = match_result["tp"], match_result["fp"], match_result["fn"]
    prf = compute_prf1(tp, fp, fn)

    return {
        "clip_name": clip_name,
        "gt_count": len(gt_events),
        "pred_count": len(pred_events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
    }


def evaluate_all(
    clips: Optional[List[str]] = None,
    predictions_dir_or_dict: Optional[Union[str, Dict[str, List[Dict[str, Any]]]]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluates metric performance across all clips. Overall Precision, Recall, and F1
    are micro-averaged (computed directly from the aggregated sum of TP, FP, and FN across
    all clips rather than macro-averaged per clip), providing a robust metric for datasets
    with imbalanced event counts.

    Saves results to:
      - outputs/eval/metrics_per_clip.csv
      - outputs/eval/metrics_overall.json
    """
    if clips is None:
        clips = ["01", "02", "03", "04", "05", "06", "07", "08"]

    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    for clip in clips:
        # Determine prediction path or list for clip
        pred_input = None
        if isinstance(predictions_dir_or_dict, dict):
            pred_input = predictions_dir_or_dict.get(clip)
        elif isinstance(predictions_dir_or_dict, str) and os.path.exists(predictions_dir_or_dict):
            possible_pred_path = os.path.join(predictions_dir_or_dict, f"pred_{clip}.csv")
            if not os.path.exists(possible_pred_path):
                possible_pred_path = os.path.join(predictions_dir_or_dict, f"pred_{clip}.json")
            if os.path.exists(possible_pred_path):
                pred_input = possible_pred_path

        gt_path = GT_FILE_MAPPING.get(clip, f"src/motion/ground_truth_{clip}.csv")
        metrics = evaluate_clip(clip, pred_csv_or_json_path=pred_input, gt_csv_path=gt_path, threshold=threshold)

        results.append(metrics)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_gt += metrics["gt_count"]
        total_pred += metrics["pred_count"]

    df = pd.DataFrame(results)

    # Compute overall micro-averaged P/R/F1 across all aggregated counts
    overall_prf = compute_prf1(total_tp, total_fp, total_fn)
    overall_summary = {
        "evaluation_type": "micro_averaged",
        "total_clips": len(clips),
        "total_gt_events": total_gt,
        "total_pred_events": total_pred,
        "sum_tp": total_tp,
        "sum_fp": total_fp,
        "sum_fn": total_fn,
        "overall_precision": overall_prf["precision"],
        "overall_recall": overall_prf["recall"],
        "overall_f1": overall_prf["f1"],
    }

    # Ensure output directory exists before writing
    out_dir = os.path.join("outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "metrics_per_clip.csv"), index=False)
    with open(os.path.join(out_dir, "metrics_overall.json"), "w") as f:
        json.dump(overall_summary, f, indent=2)

    return df


if __name__ == "__main__":
    df_results = evaluate_all()
    print("=" * 70)
    print("                EVENT EVALUATION METRICS SUMMARY TABLE              ")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)

    # Load overall metrics json for display
    overall_path = os.path.join("outputs", "eval", "metrics_overall.json")
    if os.path.exists(overall_path):
        with open(overall_path) as f:
            ov = json.load(f)
        print("\nMicro-Averaged Overall Metrics:")
        print(f"  Total Ground Truth Events : {ov['total_gt_events']}")
        print(f"  Total Predicted Events    : {ov['total_pred_events']}")
        print(f"  TP: {ov['sum_tp']} | FP: {ov['sum_fp']} | FN: {ov['sum_fn']}")
        print(f"  Precision : {ov['overall_precision']:.4f}")
        print(f"  Recall    : {ov['overall_recall']:.4f}")
        print(f"  Micro F1  : {ov['overall_f1']:.4f}")
    print("=" * 70)
