# src/track_det/overlay_frame_range.py
"""
Draws ROI boxes onto frames sampled at a fixed step across a range —
for dense visual sampling (e.g. checking for a second person across an
entire labeled window, not just the highest-fragmentation frames).

Usage:
    uv run python -m src.track_det.overlay_frame_range \
        data/real_footage/subject1/subject1_rois.csv \
        data/real_footage/subject1/frames \
        --start 3750 --end 5400 --step 40 \
        --out-dir outputs/overlays_subject1_dense
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def load_rois(csv_path):
    rows_by_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            rows_by_frame[fi].append(box)
    return rows_by_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("frames_dir")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--step", type=int, default=40,
                     help="Sample every N frames (default 40, ~1.6s at 25fps)")
    ap.add_argument("--out-dir", default="outputs/overlays_dense")
    args = ap.parse_args()

    rows_by_frame = load_rois(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)

    sample_frames = list(range(args.start, args.end + 1, args.step))
    print(f"Sampling {len(sample_frames)} frames from [{args.start}, {args.end}] every {args.step} frames.")

    written = 0
    for fi in sample_frames:
        img_path = frames_dir / f"{fi:06d}.jpg"
        if not img_path.exists():
            print(f"MISSING: {img_path} — skipping")
            continue

        img = cv2.imread(str(img_path))
        boxes = rows_by_frame.get(fi, [])

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, str(i), (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = out_dir / f"overlay_{fi:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        written += 1

    print(f"\nWrote {written} overlay images to {out_dir}/")
    print("Open these in sequence and look specifically for a second person "
          "appearing anywhere in the visible frame area, even briefly.")


if __name__ == "__main__":
    main()