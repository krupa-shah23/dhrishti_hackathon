"""
test_detect_real_footage.py

Runs detect_objects() on every frame in a real_footage frames/ folder
and reports detection rate + confidence stats, in the same style as
the webcam test (Master Doc SS4) — but on actual target-domain footage
for the first time.

Usage:
    uv run python -m src.track_det.test_detect_real_footage data/real_footage/01_001/frames
    uv run python -m src.track_det.test_detect_real_footage data/real_footage/01_0015/frames --quiet
    uv run python -m src.track_det.test_detect_real_footage data/real_footage/01_001/frames --exam-mode paper_pen
"""
import argparse
from pathlib import Path
import cv2

from .detector import detect_objects


def run(frames_dir, quiet=False, log_csv=None, exam_mode="CBT"):
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

        # detect_objects() now takes a WINDOW of crops (roi_crops_over_window,
        # exam_mode), not a single frame. This script's original intent was a
        # per-frame detection-rate/confidence report across a whole clip, so
        # each frame is wrapped as its own 1-crop window rather than batching
        # the clip into one window -- that preserves per-frame granularity but
        # means this isn't exercising real multi-frame window reduction (see
        # detect_objects()'s docstring: P1 hands it a window, never single
        # frames -- that's simulated here as the degenerate 1-frame case).
        detections = detect_objects([frame], exam_mode)

        if detections:
            frames_with_detection += 1
            for det in detections:
                cls_name, conf = det["class"], det["confidence"]
                all_confidences.append(conf)
                # No box in the frozen contract's output anymore (just
                # {"class", "confidence"}) -- log what's actually available.
                csv_rows.append((path.name, cls_name, conf))
                if not quiet:
                    print(f"Frame {i} ({path.name}): {cls_name} conf={conf:.3f}")
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
            writer.writerow(["frame_file", "class", "confidence"])
            writer.writerows(csv_rows)
        print(f"\nPer-detection log written to {log_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-csv", default=None,
                         help="optional path to write a per-detection CSV log")
    parser.add_argument("--exam-mode", default="CBT",
                         help="exam_mode passed to detect_objects() (default: CBT, "
                              "detects both phone and paper-chit)")
    args = parser.parse_args()
    run(args.frames_dir, args.quiet, args.log_csv, args.exam_mode)