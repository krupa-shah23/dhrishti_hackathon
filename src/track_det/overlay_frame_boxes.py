# src/track_det/overlay_frame_boxes.py
"""
Draws ROI boxes from subject1_rois.csv onto the actual frame image, for
visual confirmation of clustering/fragmentation findings.

Usage:
    uv run python -m src.track_det.overlay_frame_boxes \
        data/real_footage/subject1/subject1_rois.csv \
        data/real_footage/subject1/frames \
        --frames 4449 4451 4446 4450 \
        --out-dir outputs/overlays_subject1
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

            # ROI CSV
            if "frame_index" in row:
                fi = int(row["frame_index"])

            # Detection CSV
            elif "frame_file" in row:
                fi = int(
                    Path(row["frame_file"]).stem.replace("overlay_", "")
                )

            else:
                raise ValueError(
                    f"Unsupported CSV format. Columns: {reader.fieldnames}"
                )

            box = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )

            rows_by_frame[fi].append(box)

    return rows_by_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("frames_dir")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs/overlays")
    args = ap.parse_args()

    rows_by_frame = load_rois(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)

    for fi in args.frames:
        img_path = frames_dir / f"{fi:06d}.jpg"
        if not img_path.exists():
            print(f"MISSING: {img_path} — check the frame naming pattern matches your extraction.")
            continue

        img = cv2.imread(str(img_path))
        boxes = rows_by_frame.get(fi, [])

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, str(i), (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = out_dir / f"overlay_{fi:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"frame {fi}: {len(boxes)} boxes drawn -> {out_path}")


if __name__ == "__main__":
    main()