"""
test_detect_real_footage.py

Runs detect_objects() on every frame in a real_footage frames/ folder
and reports detection rate + confidence stats, in the same style as
the webcam test (Master Doc SS4) — but on actual target-domain footage
for the first time.

Usage:
    uv run python -m src.track_det.test_detect_real_footage data/real_footage/01_001/frames
    uv run python -m src.track_det.test_detect_real_footage data/real_footage/01_0015/frames --quiet
"""
import argparse
from pathlib import Path
import cv2

from .detector import detect_objects


def run(frames_dir, quiet=False, log_csv=None):
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.iterdir())
    frame_paths = [p for p in frame_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not frame_paths:
        print(f"No image files found in {frames_dir}")
        return

    total = len(frame_paths)
    frames_with_detection = 0
    all_confidences = []
    csv_rows = []

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  WARNING: could not read {path}, skipping")
            continue

        detections = detect_objects(frame)

        if detections:
            frames_with_detection += 1
            for box, cls_name, conf in detections:
                all_confidences.append(conf)
                csv_rows.append((path.name, cls_name, conf, *box))
                if not quiet:
                    print(f"Frame {i} ({path.name}): {cls_name} conf={conf:.3f} box={box}")
        else:
            if not quiet:
                print(f"Frame {i} ({path.name}): no detection")

    print(f"\n--- Summary: {frames_dir} ---")
    print(f"Total frames: {total}")
    print(f"Frames with >=1 detection: {frames_with_detection} "
          f"({100*frames_with_detection/total:.1f}%)")
    if all_confidences:
        print(f"Total detections: {len(all_confidences)}")
        print(f"Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
        print(f"Mean confidence: {sum(all_confidences)/len(all_confidences):.3f}")
    else:
        print("No detections at all across the clip.")

    if log_csv:
        import csv as csv_module
        with open(log_csv, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(["frame_file", "class", "confidence", "x1", "y1", "x2", "y2"])
            writer.writerows(csv_rows)
        print(f"\nPer-detection log written to {log_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-csv", default=None,
                         help="optional path to write a per-detection CSV log")
    args = parser.parse_args()
    run(args.frames_dir, args.quiet, args.log_csv)