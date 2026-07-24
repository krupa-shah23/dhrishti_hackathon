"""
test_roi.py
Owner: P1 - Motion & ROI

Runs motion + ROI together on each clip, saves:
- overlay sample frames (boxes drawn on original frames) for visual check
- a CSV per clip: frame_index, x1, y1, x2, y2 — one row per detected ROI,
  ready to hand off to P2's tracker.
"""

import os
import csv
import cv2
import numpy as np

from motion import MotionEstimator
from roi import get_rois, draw_rois

CLIP_PATHS = {
    "cubicle": "data/shadow/cubicle/input",
    "backdoor": "data/shadow/backdoor/input",
    "copyMachine": "data/shadow/copyMachine/input",
    "tramCrossroad_1fps": "data/lowFramerate/tramCrossroad_1fps/input",
    "fountain02": "data/dynamicBackground/fountain02/input",
    "badminton": "data/cameraJitter/badminton/input",
    "boulevard": "data/cameraJitter/boulevard/input",
    "sidewalk": "data/cameraJitter/sidewalk/input",
    "traffic": "data/cameraJitter/traffic/input",
}

MASK_OUT_DIR = "outputs/masks"
LOG_OUT_DIR = "outputs/benchmark_logs"
SAMPLE_EVERY_N_FRAMES = 20


def load_frames_from_folder(folder_path):
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".bmp")))
    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            yield fname, frame


def run_roi_check(clip_name, folder_path, exclusion_regions=None, suffix="",
                   use_stabilization=False):
    """
    Runs ROI detection on a clip and writes overlay samples + a CSV.

    exclusion_regions: list of (x1,y1,x2,y2) tuples passed straight to
        get_rois(); pass [] or None to run without any exclusion mask.
    suffix: appended to output directory name and CSV filename so that
        before/after runs don't overwrite each other (e.g. '_no_mask',
        '_with_mask').
    """
    label = clip_name + suffix
    print(f"\n--- Running ROI check on: {label} ---")
    if not os.path.isdir(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}")
        return None

    out_dir = os.path.join(MASK_OUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOG_OUT_DIR, exist_ok=True)

    estimator = MotionEstimator(use_stabilization=use_stabilization)
    csv_path = os.path.join(LOG_OUT_DIR, f"{label}_rois.csv")
    total_boxes = 0
    frame_count = 0
    shifts = []  # (dx, dy) per frame, only meaningful when use_stabilization=True

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "x1", "y1", "x2", "y2"])

        for i, (fname, original_frame) in enumerate(load_frames_from_folder(folder_path)):
            mask = estimator.get_motion_mask(original_frame)
            save_sample = (i % SAMPLE_EVERY_N_FRAMES == 0)

            if use_stabilization:
                shifts.append(estimator.last_shift)

            if save_sample:
                boxes, cleaned = get_rois(mask, return_cleaned=True,
                                          exclusion_regions=exclusion_regions)
            else:
                boxes = get_rois(mask, exclusion_regions=exclusion_regions)
                cleaned = None

            total_boxes += len(boxes)
            frame_count += 1

            for (x1, y1, x2, y2) in boxes:
                writer.writerow([i, x1, y1, x2, y2])

            if save_sample:
                overlay = draw_rois(original_frame, boxes)
                cv2.imwrite(os.path.join(out_dir, f"roi_sample_{i:05d}.jpg"), overlay)
                cv2.imwrite(os.path.join(out_dir, f"cleaned_mask_{i:05d}.png"), cleaned)

                if use_stabilization:
                    dx, dy = estimator.last_shift
                    h, w = original_frame.shape[:2]
                    translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    stabilized_frame = cv2.warpAffine(
                        original_frame, translation_matrix, (w, h),
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    diff = cv2.absdiff(original_frame, stabilized_frame)
                    cv2.imwrite(os.path.join(out_dir, f"original_{i:05d}.jpg"), original_frame)
                    cv2.imwrite(os.path.join(out_dir, f"stabilized_{i:05d}.jpg"), stabilized_frame)
                    cv2.imwrite(os.path.join(out_dir, f"diff_{i:05d}.jpg"), diff)

    avg = total_boxes / frame_count if frame_count else 0
    print(f"  Frames processed : {frame_count}")
    print(f"  Total ROIs detected : {total_boxes}")
    print(f"  Avg ROIs/frame : {avg:.3f}")
    print(f"  ROI overlay samples : {out_dir}")
    print(f"  ROI CSV (P2 handoff) : {csv_path}")

    result = {"clip": label, "frames": frame_count,
              "total_rois": total_boxes, "avg_rois": avg}

    if use_stabilization and shifts:
        mags = [ (dx**2 + dy**2) ** 0.5 for dx, dy in shifts ]
        avg_shift = sum(mags) / len(mags)
        max_shift = max(mags)
        print(f"  Avg translation (px) : {avg_shift:.2f}")
        print(f"  Max translation (px) : {max_shift:.2f}")
        result["avg_shift"] = avg_shift
        result["max_shift"] = max_shift

    return result


def compare_exclusion_mask(clip_name, folder_path, exclusion_regions):
    """
    Runs the same clip twice — once without and once with the exclusion mask —
    and prints a side-by-side summary so we can measure impact.
    """
    print(f"\n{'='*60}")
    print(f"  EXCLUSION MASK COMPARISON: {clip_name}")
    print(f"{'='*60}")

    before = run_roi_check(clip_name, folder_path,
                           exclusion_regions=None,
                           suffix="_no_mask")
    after  = run_roi_check(clip_name, folder_path,
                           exclusion_regions=exclusion_regions,
                           suffix="_with_mask")

    if before and after:
        delta_total = before["total_rois"] - after["total_rois"]
        delta_avg   = before["avg_rois"]   - after["avg_rois"]
        pct         = (delta_total / before["total_rois"] * 100
                       if before["total_rois"] else 0)
        print(f"\n  --- Comparison summary ---")
        print(f"  {'Metric':<28} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Total ROIs':<28} {before['total_rois']:>10} {after['total_rois']:>10} {-delta_total:>+10}")
        print(f"  {'Avg ROIs/frame':<28} {before['avg_rois']:>10.3f} {after['avg_rois']:>10.3f} {-delta_avg:>+10.3f}")
        print(f"  {'ROI reduction':<28} {'':<10} {'':<10} {pct:>9.1f}%")
        print(f"\n  Interpretation: negative delta = fewer detections (desired for FP zones).")
        print(f"  Verify overlay images to confirm no true detections were lost.")


def compare_stabilization(clip_name, folder_path):
    """
    Runs the same clip with and without camera-shake compensation and
    prints a side-by-side summary — expected result: avg ROIs/frame drops
    sharply once stabilization is on, if the clip has vibration-driven
    full-frame false positives.
    """
    print(f"\n{'='*60}")
    print(f"  STABILIZATION COMPARISON: {clip_name}")
    print(f"{'='*60}")

    before = run_roi_check(clip_name, folder_path,
                           suffix="_unstabilized", use_stabilization=False)
    after  = run_roi_check(clip_name, folder_path,
                           suffix="_stabilized", use_stabilization=True)

    if before and after:
        delta_avg = before["avg_rois"] - after["avg_rois"]
        pct = (delta_avg / before["avg_rois"] * 100) if before["avg_rois"] else 0
        print(f"\n  --- Comparison summary ---")
        print(f"  {'Metric':<28} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Avg ROIs/frame':<28} {before['avg_rois']:>10.3f} {after['avg_rois']:>10.3f} {-delta_avg:>+10.3f}")
        if "avg_shift" in after:
            print(f"  {'Avg translation (px)':<28} {'':<10} {after['avg_shift']:>10.2f}")
            print(f"  {'Max translation (px)':<28} {'':<10} {after['max_shift']:>10.2f}")
        print(f"\n  ROI reduction: {pct:.1f}%")
        print(f"  Interpretation: large drop = shake-driven full-frame ROIs suppressed.")


def test_ignore_regions(clip_name, folder_path, regions):
    """
    Sanity check for the invigilator ignore-region hook: runs one clip with
    a fixed region blanked out and confirms motion boxes never appear inside it.
    """
    print(f"\n{'='*60}")
    print(f"  IGNORE-REGION HOOK TEST: {clip_name}")
    print(f"{'='*60}")

    estimator = MotionEstimator()
    estimator.set_ignore_regions(regions)

    violations = 0
    frame_count = 0
    rx1, ry1, rx2, ry2 = regions[0]

    for i, (fname, frame) in enumerate(load_frames_from_folder(folder_path)):
        mask = estimator.get_motion_mask(frame)
        boxes = get_rois(mask)
        frame_count += 1
        for (x1, y1, x2, y2) in boxes:
            # box center inside the ignored region = leak
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                violations += 1

    print(f"  Frames checked : {frame_count}")
    print(f"  Region tested  : {regions[0]}")
    print(f"  Boxes leaking into ignored region : {violations}")
    print(f"  Result: {'PASS' if violations == 0 else 'FAIL'}")


if __name__ == "__main__":
    from exclusion_regions import get_exclusion_regions

    print("test_roi.py started")

    # --- Day-3 experiment: before/after comparison on copyMachine only -------
    cm_path = CLIP_PATHS["copyMachine"]
    cm_excl = get_exclusion_regions("copyMachine")
    compare_exclusion_mask("copyMachine", cm_path, cm_excl)

    for clip_name in ["badminton", "boulevard", "sidewalk", "traffic"]:
        compare_stabilization(clip_name, CLIP_PATHS[clip_name])

    test_ignore_regions("cubicle", CLIP_PATHS["cubicle"], [(0, 0, 150, 480)])

    # --- Standard run for all other clips (no exclusion mask yet) ------------
    print(f"\n{'='*60}")
    print("  STANDARD RUN (all clips, no exclusion mask)")
    print(f"{'='*60}")
    for name, path in CLIP_PATHS.items():
        if name == "copyMachine":
            # Run with the validated mask so the saved CSV is the clean version
            run_roi_check(name, path,
                          exclusion_regions=get_exclusion_regions(name))
        else:
            run_roi_check(name, path)