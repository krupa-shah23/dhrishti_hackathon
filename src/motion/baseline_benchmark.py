"""
baseline_benchmark.py
Owner: P1 - Motion & ROI
Day 5 — Benchmarks + Final Freeze

Runs 4 pipeline variants against CDNet ground truth:

  1. Frame-diff-only  — naive |frame_t - frame_t-1| threshold
  2. MOG2-only        — MOG2 mask, light cleanup only (no dilation, no
                         area filter) so it isn't artificially inflated
  3. MOG2 + ROI        — MOG2 -> morphology -> contour/area filter (roi.py)
  4. Full pipeline     — MOG2 + ROI + stabilization + exclusion regions

Reports TWO metrics, deliberately:

  (a) Pixel-level P/R/F1 — valid ONLY between frame_diff and mog2_only,
      since both output raw masks. NOT valid for mog2_roi/full, because
      roi.py intentionally dilates boxes to fix contour fragmentation
      (see Day-3 work) — a wider box is correct behavior for tracking
      but looks like "worse" pixel overlap against a pixel-tight GT mask.
      This metric would misleadingly punish exactly the fix that made
      detection more usable, so it's reported for baselines only.

  (b) Box-level P/R/F1 (IoU >= IOU_THRESHOLD, greedy match) — the fair,
      correct metric for ALL FOUR variants, since the system's real
      output (and what P2's tracker consumes) is boxes, not per-pixel
      masks. GT boxes are extracted from the GT mask via contours so
      the comparison is apples-to-apples. This is the metric that
      should be quoted as "the" benchmark result.

CDNet2014 ground truth convention (per clip's groundtruth/ folder):
    0   = background
    255 = foreground (motion)
    50  = shadow (excluded from evaluation)
    170 = unknown / don't-care region (excluded from evaluation)

Usage:
    python baseline_benchmark.py --clips data/shadow/cubicle data/shadow/backdoor --out outputs/benchmark_logs/day5_benchmark.csv
"""

import os
import csv
import argparse
import cv2
import numpy as np

try:
    from .motion import MotionEstimator, fuse_motion_signal
    from .roi import get_rois as _get_rois
    def get_rois(*args, **kwargs):
        res = _get_rois(*args, **kwargs)
        if kwargs.get('return_cleaned'):
            return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
        return [b['bbox'] if isinstance(b, dict) else b for b in res]
    from .exclusion_regions import get_exclusion_regions
except ImportError:
    from motion import MotionEstimator, fuse_motion_signal
    from roi import get_rois as _get_rois
    def get_rois(*args, **kwargs):
        res = _get_rois(*args, **kwargs)
        if kwargs.get('return_cleaned'):
            return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
        return [b['bbox'] if isinstance(b, dict) else b for b in res]
    from exclusion_regions import get_exclusion_regions
    
IOU_THRESHOLD = 0.3
GT_MIN_AREA = 200          # drop tiny GT contour noise (compression artifacts in GT itself)
BASELINE_MIN_AREA = 200    # light area filter for frame_diff/mog2_only, kept small on purpose


# ---------- I/O helpers ----------

def load_frame_list(clip_path):
    input_dir = os.path.join(clip_path, "input")
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    )
    return input_dir, files


def load_gt_mask(clip_path, gt_filename, target_shape):
    gt_dir = os.path.join(clip_path, "groundtruth")
    gt_path = os.path.join(gt_dir, gt_filename)
    if not os.path.exists(gt_path):
        return None

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None

    if gt.shape[:2] != target_shape[:2]:
        gt = cv2.resize(gt, (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    fg = (gt == 255)
    ignore = (gt == 50) | (gt == 170)
    return fg, ignore


def gt_filename_for_frame(input_filename):
    num = "".join(ch for ch in input_filename if ch.isdigit())
    return f"gt{num}.png"


# ---------- pixel-level metrics (baselines only) ----------

def pixel_prf1(pred_mask, gt_fg, gt_ignore):
    pred = pred_mask.astype(bool)
    valid = ~gt_ignore

    tp = np.count_nonzero(pred & gt_fg & valid)
    fp = np.count_nonzero(pred & ~gt_fg & valid)
    fn = np.count_nonzero(~pred & gt_fg & valid)
    tn = np.count_nonzero(~pred & ~gt_fg & valid)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "fp_rate": fp_rate}


# ---------- box-level metrics (all variants, the fair comparison) ----------

def mask_to_boxes_light(mask, min_area=BASELINE_MIN_AREA):
    """
    Minimal cleanup for baseline masks (frame_diff, mog2_only): small open
    to kill single-pixel noise, area filter, no dilation. Kept deliberately
    light so these baselines aren't given roi.py's fragmentation fix for free.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = mask.shape[:2]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, min(x + w, w_img - 1), min(y + h, h_img - 1)))
    return boxes


def gt_mask_to_boxes(gt_fg, min_area=GT_MIN_AREA):
    gt_uint8 = (gt_fg.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gt_fg.shape[:2]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, min(x + w, w_img - 1), min(y + h, h_img - 1)))
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD):
    """
    Greedy IoU matching: each GT box can be matched at most once.
    Returns (tp, fp, fn) counts for this frame.
    """
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou(pb, gb)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_threshold:
            matched_gt.add(best_j)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ---------- pipeline variants ----------

def variant_frame_diff(prev_frame, frame, threshold=25):
    if prev_frame is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    diff = cv2.absdiff(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def variant_mog2_only(estimator, frame):
    _, mask = estimator.get_motion_mask(frame)
    return mask


def variant_mog2_roi(estimator, frame, exclusion_regions=None):
    mag_map, mask = estimator.get_motion_mask(frame)
    return get_rois(mask, exclusion_regions=exclusion_regions), mask


def variant_full_pipeline(estimator, frame, exclusion_regions=None):
    mag_map, mask = estimator.get_motion_mask(frame)  # estimator has use_stabilization=True
    fused_mask = fuse_motion_signal(mag_map, mask)
    return get_rois(fused_mask, exclusion_regions=exclusion_regions), fused_mask


# ---------- benchmark runner ----------

def run_benchmark(clip_path, clip_name):
    input_dir, files = load_frame_list(clip_path)
    exclusion_regions = EXCLUSION_REGIONS.get(clip_name, [])

    fd_prev_frame = None
    est_mog2only = MotionEstimator(use_stabilization=False)
    est_roi = MotionEstimator(use_stabilization=False)
    est_full = MotionEstimator(use_stabilization=True)

    # pixel accum: baselines only
    pixel_accum = {
        "frame_diff": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "mog2_only":  {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    # box accum: all four variants
    box_accum = {
        "frame_diff": {"tp": 0, "fp": 0, "fn": 0},
        "mog2_only":  {"tp": 0, "fp": 0, "fn": 0},
        "mog2_roi":   {"tp": 0, "fp": 0, "fn": 0},
        "full":       {"tp": 0, "fp": 0, "fn": 0},
    }
    frames_scored = 0

    for fname in files:
        frame = cv2.imread(os.path.join(input_dir, fname))
        if frame is None:
            continue

        gt_result = load_gt_mask(clip_path, gt_filename_for_frame(fname), frame.shape)

        mask_fd = variant_frame_diff(fd_prev_frame, frame)
        mask_mog2 = variant_mog2_only(est_mog2only, frame)
        boxes_roi, _ = variant_mog2_roi(est_roi, frame, exclusion_regions)
        boxes_full, _ = variant_full_pipeline(est_full, frame, exclusion_regions)
        fd_prev_frame = frame

        if gt_result is None:
            continue

        gt_fg, gt_ignore = gt_result
        frames_scored += 1

        # pixel-level, baselines only
        for name, mask in [("frame_diff", mask_fd), ("mog2_only", mask_mog2)]:
            r = pixel_prf1(mask, gt_fg, gt_ignore)
            # re-derive raw counts from rates isn't possible; recompute directly
        # (recompute pixel tp/fp/fn/tn directly, not from prf1 dict, to accumulate correctly)
        for name, mask in [("frame_diff", mask_fd), ("mog2_only", mask_mog2)]:
            pred = mask.astype(bool)
            valid = ~gt_ignore
            pixel_accum[name]["tp"] += np.count_nonzero(pred & gt_fg & valid)
            pixel_accum[name]["fp"] += np.count_nonzero(pred & ~gt_fg & valid)
            pixel_accum[name]["fn"] += np.count_nonzero(~pred & gt_fg & valid)
            pixel_accum[name]["tn"] += np.count_nonzero(~pred & ~gt_fg & valid)

        # box-level, all four variants
        gt_boxes = gt_mask_to_boxes(gt_fg)
        boxes_fd = mask_to_boxes_light(mask_fd)
        boxes_mog2 = mask_to_boxes_light(mask_mog2)

        for name, pred_boxes in [("frame_diff", boxes_fd), ("mog2_only", boxes_mog2),
                                  ("mog2_roi", boxes_roi), ("full", boxes_full)]:
            tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
            box_accum[name]["tp"] += tp
            box_accum[name]["fp"] += fp
            box_accum[name]["fn"] += fn

    # finalize pixel metrics
    pixel_results = {}
    for name, c in pixel_accum.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        pixel_results[name] = {"precision": precision, "recall": recall, "f1": f1}

    # finalize box metrics
    box_results = {}
    for name, c in box_accum.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        box_results[name] = {"precision": precision, "recall": recall, "f1": f1}

    return pixel_results, box_results, frames_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for clip_path in args.clips:
        clip_name = os.path.basename(os.path.normpath(clip_path))
        print(f"\n[+] Benchmarking {clip_name}...")
        pixel_results, box_results, n = run_benchmark(clip_path, clip_name)
        print(f"    Frames scored: {n}")

        print(f"    -- pixel-level (frame_diff / mog2_only only) --")
        for name in ["frame_diff", "mog2_only"]:
            m = pixel_results[name]
            print(f"    {name:<12} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

        print(f"    -- box-level IoU>={IOU_THRESHOLD} (all variants — the fair comparison) --")
        for name in ["frame_diff", "mog2_only", "mog2_roi", "full"]:
            m = box_results[name]
            print(f"    {name:<12} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

        for name in ["frame_diff", "mog2_only", "mog2_roi", "full"]:
            row = {"clip": clip_name, "variant": name, "frames_scored": n,
                   "box_precision": box_results[name]["precision"],
                   "box_recall": box_results[name]["recall"],
                   "box_f1": box_results[name]["f1"]}
            if name in pixel_results:
                row["pixel_precision"] = pixel_results[name]["precision"]
                row["pixel_recall"] = pixel_results[name]["recall"]
                row["pixel_f1"] = pixel_results[name]["f1"]
            else:
                row["pixel_precision"] = row["pixel_recall"] = row["pixel_f1"] = ""
            rows.append(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "variant", "frames_scored",
            "box_precision", "box_recall", "box_f1",
            "pixel_precision", "pixel_recall", "pixel_f1",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Benchmark table written to {args.out}")


if __name__ == "__main__":
    main()