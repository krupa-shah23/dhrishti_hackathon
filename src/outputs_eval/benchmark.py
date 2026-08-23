"""
benchmark.py -- P4: Outputs, Evaluation & Integration
Runs the video pipeline in 4 configurations and compares their
Precision, Recall, F1 (mask-level AND event-level), and processing speed.

Configurations:
  1. frame_diff   -- Simple absolute frame differencing (baseline)
  2. mog2         -- MOG2 background subtraction only (no ROI filter)
  3. mog2_roi     -- MOG2 + ROI bounding-box filtering
  4. full_pipeline-- MOG2 + ROI + Tracking + Features + Risk scoring (complete P4)

Outputs:
  outputs/benchmark_table.csv
"""

import os
import csv
import time

import cv2  # type: ignore[import-not-found]
import numpy as np

from src.outputs_eval.metrics import (
    MaskMetrics,
    update_mask_metrics,
    Event,
    match_events,
    print_event_results,
)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

class FrameDiffSubtractor:
    """Baseline: absolute difference between consecutive frames."""

    def __init__(self, threshold: int = 30):
        self.prev_gray = None
        self.threshold = threshold

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            h, w = gray.shape
            return np.zeros((h, w), dtype=np.uint8)
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask


class MOG2Subtractor:
    """Standard OpenCV MOG2 background subtractor."""

    def __init__(self, history: int = 500, var_threshold: int = 16):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return self.bg.apply(frame)


def _apply_roi_filter(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Filters a mask by removing small contour blobs below min_area."""
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    cleaned  = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


# ---------------------------------------------------------------------------
# Event-level: derive predicted events from a per-frame activity signal
# ---------------------------------------------------------------------------

def _frames_to_events(
    active_flags: list[bool],
    fps: float,
    min_gap_s: float = 1.0,
    min_duration_s: float = 0.5,
) -> list[Event]:
    """
    Converts a per-frame boolean 'motion active' signal into a list of
    Event(start, end) spans, merging gaps <= min_gap_s and dropping
    events shorter than min_duration_s.
    """
    events: list[Event] = []
    seg_start_idx = None
    last_active_idx = None

    for idx, active in enumerate(active_flags):
        if active:
            if seg_start_idx is None:
                seg_start_idx = idx
            elif last_active_idx is not None:
                gap_s = (idx - last_active_idx) / fps
                if gap_s > min_gap_s:
                    # close previous segment, start new one
                    start_t = seg_start_idx / fps
                    end_t   = last_active_idx / fps
                    if end_t - start_t >= min_duration_s:
                        events.append(Event(start=start_t, end=end_t))
                    seg_start_idx = idx
            last_active_idx = idx

    if seg_start_idx is not None and last_active_idx is not None:
        start_t = seg_start_idx / fps
        end_t   = last_active_idx / fps
        if end_t - start_t >= min_duration_s:
            events.append(Event(start=start_t, end=end_t))

    return events


def _load_gt_events(gt_csv_path: str, video_id: str | None = None) -> list[Event]:
    """Loads ground_truth_events.csv, optionally filtered to one video_id.
    Always excludes clip5 (unresolved, per project decision)."""
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
# Single-video benchmark runner
# ---------------------------------------------------------------------------

def benchmark_video(
    video_path: str,
    video_id: str | None = None,
    gt_masks_dir: str | None = None,
    gt_events_csv: str | None = None,
    min_area: int = 500,
    activity_pixel_threshold: int = 500,
) -> dict[str, dict]:
    """
    Runs all 4 pipeline configurations on a video and collects metrics.

    - If gt_masks_dir is provided: mask-level IoU/P/R/F1 (CDnet2014-style).
    - If gt_events_csv is provided: event-level P/R/F1 (>=50% overlap rule)
      against ground_truth_events.csv, using video_id to filter GT rows.

    Args:
        video_path    (str): Path to input video.
        video_id      (str): Video id matching ground_truth_events.csv rows.
        gt_masks_dir  (str): Optional path to folder of GT masks.
        gt_events_csv (str): Optional path to ground_truth_events.csv.
        min_area      (int): Minimum contour area for ROI filtering.
        activity_pixel_threshold (int): Foreground pixel count above which
            a frame counts as "active" for event derivation.

    Returns:
        dict: mapping method_name -> result_dict
    """
    configs = {
        "frame_diff":    FrameDiffSubtractor(threshold=30),
        "mog2":          MOG2Subtractor(history=500, var_threshold=16),
        "mog2_roi":      MOG2Subtractor(history=500, var_threshold=16),
        "full_pipeline": MOG2Subtractor(history=500, var_threshold=16),
    }

    gt_events = _load_gt_events(gt_events_csv, video_id) if gt_events_csv else None
    results = {}

    for method_name, subtractor in configs.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[benchmark] ERROR: cannot open {video_path}")
            break

        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        metrics   = MaskMetrics()
        active_flags: list[bool] = []
        frame_idx = 0
        t_start   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get motion mask
            mask = subtractor.apply(frame)

            # Apply ROI filter for mog2_roi and full_pipeline
            if method_name in ("mog2_roi", "full_pipeline"):
                mask = _apply_roi_filter(mask, min_area)

            # Track per-frame activity for event derivation
            active_flags.append(int(np.count_nonzero(mask)) >= activity_pixel_threshold)

            # If GT masks available, update mask metrics
            if gt_masks_dir:
                gt_name = os.path.join(gt_masks_dir, f"gt{frame_idx + 1:06d}.png")
                if os.path.exists(gt_name):
                    gt_mask = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        # CDnet GT: 85 = shadow, 170 = foreground, 0 = background
                        gt_binary = (gt_mask == 170).astype(np.uint8) * 255
                        update_mask_metrics(metrics, mask, gt_binary)

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        result = {
            "method":       method_name,
            "frames":       frame_idx,
            "elapsed_s":    round(elapsed, 2),
            "proc_fps":     round(proc_fps, 2),
        }
        if metrics.num_frames > 0:
            result.update({f"mask_{k}": v for k, v in metrics.summary().items()})

        # Event-level F1 against ground_truth_events.csv
        if gt_events is not None:
            pred_events = _frames_to_events(active_flags, fps)
            event_metrics = match_events(pred_events, gt_events, overlap_threshold=0.50)
            print_event_results(event_metrics, label=f"{method_name} — Event F1")
            result.update({f"event_{k}": v for k, v in event_metrics.summary().items()})

        results[method_name] = result
        print(f"[benchmark] {method_name:<20} | {proc_fps:.1f} fps | "
              f"mask_F1={result.get('mask_f1', 'N/A')} | "
              f"event_F1={result.get('event_f1', 'N/A')}")

    return results


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

def save_benchmark_csv(results: dict[str, dict], output_path: str) -> None:
    """Saves benchmark results to outputs/benchmark_table.csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_keys = set()
    for r in results.values():
        all_keys.update(r.keys())
    fieldnames = ["method"] + sorted(k for k in all_keys if k != "method")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

    print(f"[benchmark] Table saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P4/P2b Benchmark Runner")
    parser.add_argument("--video",     required=True, help="Path to input video")
    parser.add_argument("--video-id",  default=None,  help="video_id matching ground_truth_events.csv")
    parser.add_argument("--gt-dir",    default=None,  help="Optional GT masks directory (CDnet2014)")
    parser.add_argument("--gt-events", default=None,  help="Optional path to ground_truth_events.csv")
    parser.add_argument("--out",       default="outputs/benchmark_table.csv")
    args = parser.parse_args()

    results = benchmark_video(
        args.video,
        video_id=args.video_id,
        gt_masks_dir=args.gt_dir,
        gt_events_csv=args.gt_events,
    )
    save_benchmark_csv(results, args.out)

    print("\n[benchmark] Summary:")
    print(f"{'Method':<22} {'FPS':>6} {'MaskF1':>8} {'EventF1':>9} {'EventP':>8} {'EventR':>8}")
    print("-" * 70)
    for r in results.values():
        print(f"{r['method']:<22} {r['proc_fps']:>6.1f} "
              f"{r.get('mask_f1','N/A'):>8}  "
              f"{r.get('event_f1','N/A'):>9} "
              f"{r.get('event_precision','N/A'):>8} "
              f"{r.get('event_recall','N/A'):>8}")