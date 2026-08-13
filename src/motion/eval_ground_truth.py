"""
eval_ground_truth.py

Computes Precision / Recall / F1 against CDNet2014 ground truth for the
motion/ROI pipeline, run both with and without shake stabilization.

Two metrics are reported per clip:
  - Pixel-level F1: cleaned binary mask vs gt==255, restricted to ROI, 170=don't-care
  - Box-level F1:   rasterized ROI boxes vs gt==255, same restriction

Assumes clip folder layout (standard CDNet2014):
  <clip_dir>/input/in%06d.jpg
  <clip_dir>/groundtruth/gt%06d.png
  <clip_dir>/ROI.bmp   (or ROI.jpg)   -- binary, 255 = valid eval region
  <clip_dir>/temporalROI.txt          -- "start_frame end_frame" (1-indexed, inclusive)

Usage:
  python src/motion/eval_ground_truth.py --clips path/to/badminton path/to/traffic --out outputs/eval_ground_truth.csv

Adjust the imports below to match your actual module names for:
  - get_motion_mask(frame) -> binary mask (0/255)
  - get_rois(mask) -> list of (x, y, w, h)
  - shake_compensation warp step
so this plugs directly into your existing pipeline instead of reimplementing it.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from motion import MotionEstimator
from roi import get_rois

GT_MOTION = 255
GT_DONTCARE = 170  # unknown -- excluded from scoring
GT_OUTSIDE_ROI = 85  # region outside evaluation ROI, per-frame variant some clips use


def load_roi_mask(clip_dir):
    for name in ("ROI.bmp", "ROI.jpg", "ROI.png"):
        path = os.path.join(clip_dir, name)
        if os.path.exists(path):
            roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return (roi > 127).astype(np.uint8)
    # no ROI file -> everything valid
    return None


def load_temporal_roi(clip_dir, total_frames):
    path = os.path.join(clip_dir, "temporalROI.txt")
    if not os.path.exists(path):
        return 1, total_frames
    with open(path) as f:
        start, end = map(int, f.read().split())
    return start, end


from baseline_benchmark import gt_mask_to_boxes, match_boxes  # reuse the same IoU-matched box scoring as the Day-5 benchmark table, so box-F1 means the same thing in both scripts


def accumulate_confusion(pred_bin, gt_frame, roi_mask):
    """pred_bin: 0/1 array. gt_frame: raw grayscale gt (0/50/85/170/255)."""
    valid = np.ones(gt_frame.shape, dtype=bool)
    if roi_mask is not None:
        valid &= (roi_mask == 1)
    valid &= (gt_frame != GT_DONTCARE)
    valid &= (gt_frame != GT_OUTSIDE_ROI)

    gt_pos = (gt_frame == GT_MOTION) & valid
    gt_neg = (gt_frame != GT_MOTION) & valid  # includes shadow(50)/static(0) as negative

    pred_pos = (pred_bin == 1) & valid

    tp = int(np.sum(pred_pos & gt_pos))
    fp = int(np.sum(pred_pos & gt_neg))
    fn = int(np.sum((~pred_pos) & gt_pos))
    tn = int(np.sum((~pred_pos) & gt_neg))
    return tp, fp, fn, tn


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_clip(clip_dir, use_stabilization):
    input_dir = os.path.join(clip_dir, "input")
    gt_dir = os.path.join(clip_dir, "groundtruth")
    frame_files = sorted(os.listdir(input_dir))
    gt_files = sorted(os.listdir(gt_dir))
    total_frames = len(frame_files)

    roi_mask = load_roi_mask(clip_dir)
    start, end = load_temporal_roi(clip_dir, total_frames)

    # Some CDNet clips ship ROI.bmp at a different resolution than the
    # actual frames/ground-truth (known dataset inconsistency, e.g. traffic).
    # Resize with nearest-neighbor (binary mask -- no interpolation blur).
    if roi_mask is not None and frame_files and gt_files:
        probe_gt = cv2.imread(os.path.join(gt_dir, gt_files[0]), cv2.IMREAD_GRAYSCALE)
        if probe_gt is not None and roi_mask.shape != probe_gt.shape:
            roi_mask = cv2.resize(
                roi_mask, (probe_gt.shape[1], probe_gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

    pix_tp = pix_fp = pix_fn = 0
    box_tp = box_fp = box_fn = 0

    estimator = MotionEstimator(use_stabilization=use_stabilization)

    for i, (ff, gf) in enumerate(zip(frame_files, gt_files), start=1):
        frame = cv2.imread(os.path.join(input_dir, ff))
        gt = cv2.imread(os.path.join(gt_dir, gf), cv2.IMREAD_GRAYSCALE)
        if frame is None or gt is None:
            continue
        if i < start or i > end:
            continue  # CDNet excludes bootstrap/init frames per clip

        mask = estimator.get_motion_mask(frame)  # 0/255; stabilization applied internally if enabled
        mask_bin = (mask > 127).astype(np.uint8)

        tp, fp, fn, _ = accumulate_confusion(mask_bin, gt, roi_mask)
        pix_tp += tp; pix_fp += fp; pix_fn += fn

        # Box-level: IoU-matched detection comparison, same method as baseline_benchmark.py,
        # NOT pixel-overlap of rasterized boxes (that was the earlier inconsistency).
        pred_boxes = get_rois(mask)
        gt_fg = (gt == GT_MOTION)
        if roi_mask is not None:
            gt_fg = gt_fg & (roi_mask == 1)
        gt_boxes = gt_mask_to_boxes(gt_fg)
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
        box_tp += tp; box_fp += fp; box_fn += fn

    pix_p, pix_r, pix_f1 = prf1(pix_tp, pix_fp, pix_fn)
    box_p, box_r, box_f1 = prf1(box_tp, box_fp, box_fn)
    return {
        "pixel_precision": pix_p, "pixel_recall": pix_r, "pixel_f1": pix_f1,
        "box_precision": box_p, "box_recall": box_r, "box_f1": box_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="clip directories")
    ap.add_argument("--out", default="outputs/eval_ground_truth.csv")
    args = ap.parse_args()

    rows = []
    for clip_dir in args.clips:
        name = os.path.basename(os.path.normpath(clip_dir))

        print(f"[+] Evaluating {name} (no stabilization)...")
        r_off = run_clip(clip_dir, use_stabilization=False)

        print(f"[+] Evaluating {name} (with stabilization)...")
        r_on = run_clip(clip_dir, use_stabilization=True)

        row = {"clip": name}
        for k, v in r_off.items():
            row[f"{k}_nostab"] = round(v, 4)
        for k, v in r_on.items():
            row[f"{k}_stab"] = round(v, 4)
        row["delta_pixel_f1"] = round(r_on["pixel_f1"] - r_off["pixel_f1"], 4)
        row["delta_box_f1"] = round(r_on["box_f1"] - r_off["box_f1"], 4)
        rows.append(row)

        print(f"    pixel F1: {r_off['pixel_f1']:.4f} -> {r_on['pixel_f1']:.4f}  (Δ {row['delta_pixel_f1']:+.4f})")
        print(f"    box   F1: {r_off['box_f1']:.4f} -> {r_on['box_f1']:.4f}  (Δ {row['delta_box_f1']:+.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[✓] Results written to {args.out}")


if __name__ == "__main__":
    main()