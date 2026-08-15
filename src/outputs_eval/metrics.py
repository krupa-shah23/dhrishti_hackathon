"""
metrics.py -- P4: Outputs, Evaluation & Integration
Evaluation metrics for two tasks:
  1. Mask-level evaluation against CDnet2014 ground-truth
     -> Computes per-frame IoU, then aggregates Precision, Recall, F1
  2. Event-level evaluation against manually annotated demo clips
     -> Matches predicted events to GT events with >=50% temporal overlap criterion
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class MaskMetrics:
    """Aggregate results of frame-level mask evaluation (CDnet2014)."""
    num_frames: int = 0
    total_iou: float = 0.0
    tp: int = 0       # true positive pixels
    fp: int = 0       # false positive pixels
    fn: int = 0       # false negative pixels

    @property
    def mean_iou(self) -> float:
        return self.total_iou / self.num_frames if self.num_frames else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {
            "frames_evaluated": self.num_frames,
            "mean_iou":  round(self.mean_iou,  4),
            "precision": round(self.precision,  4),
            "recall":    round(self.recall,     4),
            "f1":        round(self.f1,         4),
        }


@dataclass
class EventMetrics:
    """Aggregate results of event-level evaluation (demo clips)."""
    tp_events: int = 0
    fp_events: int = 0
    fn_events: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp_events + self.fp_events
        return self.tp_events / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp_events + self.fn_events
        return self.tp_events / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {
            "tp":        self.tp_events,
            "fp":        self.fp_events,
            "fn":        self.fn_events,
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "f1":        round(self.f1,        4),
        }


# ---------------------------------------------------------------------------
# 1. Mask-level evaluation (CDnet2014)
# ---------------------------------------------------------------------------

def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union between a predicted binary mask and
    the CDnet2014 ground-truth mask.

    Pixels with value > 0 are treated as foreground.

    Args:
        pred_mask (np.ndarray): Predicted binary mask (H, W).
        gt_mask   (np.ndarray): Ground-truth binary mask (H, W).

    Returns:
        float: IoU in range [0.0, 1.0].
    """
    pred_bool = pred_mask > 0
    gt_bool   = gt_mask   > 0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union        = np.logical_or(pred_bool,  gt_bool).sum()

    return float(intersection) / float(union) if union > 0 else 0.0


def update_mask_metrics(
    metrics: MaskMetrics,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> MaskMetrics:
    """
    Updates the running MaskMetrics accumulator with a single frame pair.

    Args:
        metrics   (MaskMetrics): Running accumulator to update in-place.
        pred_mask (np.ndarray): Predicted mask.
        gt_mask   (np.ndarray): Ground-truth mask.

    Returns:
        MaskMetrics: Updated accumulator.
    """
    pred_bool = pred_mask > 0
    gt_bool   = gt_mask   > 0

    metrics.tp += int(np.logical_and(pred_bool,  gt_bool).sum())
    metrics.fp += int(np.logical_and(pred_bool,  ~gt_bool).sum())
    metrics.fn += int(np.logical_and(~pred_bool, gt_bool).sum())
    metrics.total_iou += mask_iou(pred_mask, gt_mask)
    metrics.num_frames += 1

    return metrics


def evaluate_masks(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> MaskMetrics:
    """
    Runs full mask-level evaluation over a list of frame pairs.

    Args:
        pred_masks: List of predicted binary masks.
        gt_masks:   List of ground-truth binary masks.

    Returns:
        MaskMetrics with aggregated IoU, P, R, F1.
    """
    assert len(pred_masks) == len(gt_masks), \
        f"Frame count mismatch: pred={len(pred_masks)}, gt={len(gt_masks)}"

    metrics = MaskMetrics()
    for pred, gt in zip(pred_masks, gt_masks):
        update_mask_metrics(metrics, pred, gt)

    return metrics


# ---------------------------------------------------------------------------
# 2. Event-level evaluation (demo clips, >=50% temporal overlap criterion)
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A temporal event defined by start and end time in seconds."""
    start: float
    end: float
    label: str = "suspicious"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def temporal_overlap_ratio(pred: Event, gt: Event) -> float:
    """
    Computes the temporal overlap ratio between a predicted and GT event
    using the intersection-over-union of their time spans.

    Args:
        pred (Event): Predicted event.
        gt   (Event): Ground-truth event.

    Returns:
        float: Temporal IoU in range [0.0, 1.0].
    """
    inter_start = max(pred.start, gt.start)
    inter_end   = min(pred.end,   gt.end)
    intersection = max(0.0, inter_end - inter_start)

    union = pred.duration + gt.duration - intersection
    return intersection / union if union > 0 else 0.0


def match_events(
    pred_events: list[Event],
    gt_events: list[Event],
    overlap_threshold: float = 0.50,
) -> EventMetrics:
    """
    Matches predicted events to ground-truth events using the >=50% temporal
    overlap criterion. Each GT event can only be matched once (greedy).

    Args:
        pred_events        (list[Event]): List of predicted events.
        gt_events          (list[Event]): List of annotated ground-truth events.
        overlap_threshold  (float):       Minimum temporal IoU to count as a match.
                                          Default: 0.50 (as per PRD spec).

    Returns:
        EventMetrics with TP, FP, FN counts and derived P, R, F1.
    """
    metrics    = EventMetrics()
    matched_gt = set()   # indices of GT events already matched

    for pred in pred_events:
        best_iou   = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_events):
            if gt_idx in matched_gt:
                continue
            iou = temporal_overlap_ratio(pred, gt)
            if iou > best_iou:
                best_iou    = iou
                best_gt_idx = gt_idx

        if best_iou >= overlap_threshold:
            metrics.tp_events += 1
            matched_gt.add(best_gt_idx)
        else:
            metrics.fp_events += 1

    # Unmatched GT events are false negatives
    metrics.fn_events = len(gt_events) - len(matched_gt)

    return metrics


# ---------------------------------------------------------------------------
# 3. Convenience: load events from the CSV log
# ---------------------------------------------------------------------------

def load_events_from_csv(csv_path: str, risk_threshold: float = 0.65) -> list[Event]:
    """
    Reads the P4 events CSV and converts high-risk rows into Event objects.
    Consecutive frames above the threshold are merged into a single event.

    Args:
        csv_path       (str):   Path to <video_id>_events.csv
        risk_threshold (float): Rows with risk_score >= threshold are included.

    Returns:
        list[Event]: Merged event list.
    """
    import csv

    raw_events: list[float] = []  # timestamp in seconds

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                risk = float(row.get("risk_score", 0))
                if risk < risk_threshold:
                    continue
                ts_str = row["timestamp"]  # HH:MM:SS.mmm
                h, m, s = ts_str.split(":")
                t = int(h) * 3600 + int(m) * 60 + float(s)
                raw_events.append(t)

    except FileNotFoundError:
        print(f"[metrics] WARNING: CSV not found at {csv_path}")
        return []

    if not raw_events:
        return []

    # Merge timestamps within 1.0s gap into a single event
    raw_events.sort()
    merged: list[Event] = []
    seg_start: float = raw_events[0]
    seg_end: float   = raw_events[0]

    for t in raw_events[1:]:
        if t - seg_end <= 1.0:
            seg_end = t
        else:
            merged.append(Event(start=seg_start, end=seg_end + 0.04))
            seg_start = t
            seg_end   = t

    merged.append(Event(start=seg_start, end=seg_end + 0.04))
    return merged


# ---------------------------------------------------------------------------
# 4. Pretty-print helpers
# ---------------------------------------------------------------------------

def print_mask_results(m: MaskMetrics, label: str = "Mask Eval") -> None:
    print(f"\n[metrics] === {label} ===")
    print(f"  Frames evaluated : {m.num_frames}")
    print(f"  Mean IoU         : {m.mean_iou:.4f}")
    print(f"  Precision        : {m.precision:.4f}")
    print(f"  Recall           : {m.recall:.4f}")
    print(f"  F1-score         : {m.f1:.4f}")


def print_event_results(e: EventMetrics, label: str = "Event Eval") -> None:
    print(f"\n[metrics] === {label} ===")
    print(f"  TP / FP / FN : {e.tp_events} / {e.fp_events} / {e.fn_events}")
    print(f"  Precision    : {e.precision:.4f}")
    print(f"  Recall       : {e.recall:.4f}")
    print(f"  F1-score     : {e.f1:.4f}")
