# src/track_det/check_detection_subject10.py
"""
Applies the same 3-check false-positive validation used on subject1
(§19c/§2a of session handoff) to subject10 -- but Check 2 is adapted,
because subject10's gt.txt has NO phone-use (label=5) window at all
(mostly label 2, some label 1). There's no known-true-positive window
to check overlap against here, so instead we check whether high-
confidence detections cluster on any particular activity label or are
scattered the same way subject1's were (which would suggest the same
false-positive mechanism -- shirt-shadow/wearcam device -- rather than
a new failure mode).

Usage:
    uv run python -m src.track_det.check_detection_subject10 \
        outputs/detections_subject10.csv \
        data/real_footage/subject10/gt.txt
"""
import sys
import csv as csv_module
from pathlib import Path
from collections import Counter


def load_detections(csv_path):
    """
    Loads detections from detections_subject10.csv.

    Expected columns:
        frame_file, class, confidence, x1, y1, x2, y2

    Converts frame_file (e.g. overlay_003750.jpg) into an integer frame
    number (3750) so the rest of the script can remain unchanged.
    """
    rows = []

    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)

        for row in reader:
            frame = int(
                Path(row["frame_file"]).stem.replace("overlay_", "")
            )

            rows.append({
                "frame": frame,
                "confidence": float(row["confidence"]),
            })

    return rows


def load_gt_labels(gt_path):
    """
    Expects gt.txt lines of the form: start_frame  end_frame  label
    (matching the format already used for subject1's label=5 window
    and label=2 "talking to a person" window elsewhere in this project).
    Returns a list of (start, end, label) tuples.
    """
    windows = []
    with open(gt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                start, end, label = int(parts[0]), int(parts[1]), int(parts[2])
                windows.append((start, end, label))
    return windows


def label_for_frame(frame, windows):
    for start, end, label in windows:
        if start <= frame <= end:
            return label
    return None  # frame not covered by any labeled window


def main():
    det_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/detections_subject10.csv"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/real_footage/subject10/gt.txt"

    detections = load_detections(det_path)
    windows = load_gt_labels(gt_path)

    total = len(detections)
    below_05 = sum(1 for d in detections if d["confidence"] < 0.5)
    high_conf = [d for d in detections if d["confidence"] >= 0.6]

    print("--- Check 1: confidence distribution ---")
    print(f"Total detections: {total}")
    print(f"Below 0.5 confidence: {below_05} ({100*below_05/total:.1f}%)")
    if detections:
        frames = [d["frame"] for d in detections]
        print(f"First detection at frame {min(frames)}, last at frame {max(frames)}")

    print("\n--- Check 2 (adapted): label distribution of high-confidence detections ---")
    print(f"Detections >= 0.6 confidence: {len(high_conf)}")
    if not windows:
        print("No gt.txt windows loaded -- cannot map detections to labels.")
    else:
        label_counts = Counter(label_for_frame(d["frame"], windows) for d in high_conf)
        print("High-confidence detections by ground-truth label at that frame:")
        for label, count in sorted(label_counts.items(), key=lambda x: (x[0] is None, x[0])):
            label_str = "no label / uncovered" if label is None else f"label={label}"
            print(f"  {label_str}: {count}")
        print(
            "\nInterpretation: since subject10 has no label=5 (phone use) window at all, "
            "any high-confidence detections here are, by definition, on non-phone-use "
            "activity. If they're concentrated on one label (e.g. label=2), that's still "
            "worth a visual check -- but if they're scattered across labels/frames the same "
            "way subject1's were, that supports the same false-positive mechanism "
            "(shirt-shadow / wearcam device) rather than a new one."
        )

    print("\n--- Check 3: strongest cluster (for visual follow-up) ---")
    if high_conf:
        by_frame = sorted(high_conf, key=lambda d: d["frame"])
        # crude contiguous-cluster finder: group detections within 30 frames of each other
        clusters = []
        current = [by_frame[0]]
        for d in by_frame[1:]:
            if d["frame"] - current[-1]["frame"] <= 30:
                current.append(d)
            else:
                clusters.append(current)
                current = [d]
        clusters.append(current)
        strongest = max(clusters, key=len)
        print(
            f"Strongest cluster: frames {strongest[0]['frame']}-{strongest[-1]['frame']} "
            f"({len(strongest)} detections), max confidence "
            f"{max(d['confidence'] for d in strongest):.3f}"
        )
        print(
            "Next step: pull overlay frames across this range (reuse "
            "overlay_frame_boxes.py, same as subject1's frames 16572-16683 check) "
            "and inspect visually for a phone/chit before drawing any conclusion."
        )
    else:
        print("No detections >= 0.6 confidence -- nothing to visually check.")


if __name__ == "__main__":
    main()