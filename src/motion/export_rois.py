"""
export_rois.py
Owner: P1 - Motion & ROI

Exports per-frame ROI detections to CSV for downstream integration
(e.g. P2's tracker consuming boxes directly rather than via the live pipeline).

Uses the existing pipeline exactly as-is, via the shared contract functions:
    get_motion_mask(frame) -> mask      (motion.py)
    get_rois(mask) -> [(x1, y1, x2, y2), ...]   (roi.py)

No existing files are modified. This script is read-only with respect to
the pipeline modules.

Input is a CDNet-style image-sequence folder (e.g. data/shadow/cubicle/input),
matching how the rest of P1 already reads clips (see motion.py:process_clip,
baseline_benchmark.py:load_frame_list) — not a video container.

Output: src/p1_rois.csv
    header: frame_index,x1,y1,x2,y2
    one row per ROI. Frames with zero ROIs are omitted entirely (no row).

Usage:
    python src/motion/export_rois.py --input_dir data/shadow/cubicle/input
    python src/motion/export_rois.py --input_dir data/shadow/cubicle/input --out src/p1_rois.csv
"""

import argparse
import csv
import os

import cv2

from motion import get_motion_mask
from roi import get_rois

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "p1_rois.csv")


def load_frame_files(input_dir):
    """Sorted list of image filenames in the folder (jpg/png/bmp), same
    convention used by motion.py:process_clip and baseline_benchmark.py."""
    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    )


def export_rois(input_dir, out_path):
    files = load_frame_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No frame images found in: {input_dir}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    width = height = None
    frame_index = 0
    total_rois = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "x1", "y1", "x2", "y2"])

        for fname in files:
            frame = cv2.imread(os.path.join(input_dir, fname))
            if frame is None:
                continue

            if width is None:
                height, width = frame.shape[:2]

            mask = get_motion_mask(frame)
            boxes = get_rois(mask)

            for (x1, y1, x2, y2) in boxes:
                writer.writerow([frame_index, x1, y1, x2, y2])
                total_rois += 1

            frame_index += 1

    return {
        "width": width,
        "height": height,
        "fps": "N/A (image sequence)",
        "frames_processed": frame_index,
        "total_rois": total_rois,
        "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                     help="path to CDNet-style input frame folder, e.g. data/shadow/cubicle/input")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path (default: src/p1_rois.csv)")
    args = ap.parse_args()

    stats = export_rois(args.input_dir, args.out)

    print(f"Resolution:       {stats['width']}x{stats['height']}")
    print(f"FPS:              {stats['fps']}")
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"ROIs exported:    {stats['total_rois']}")
    print(f"Output file:      {stats['out_path']}")


if __name__ == "__main__":
    main()