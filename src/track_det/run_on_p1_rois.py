"""
Loads P1's ROI boxes from CSV and feeds them into track() one frame at a
time, in order — matching track()'s expectation of sequential per-frame
calls with internal state carried between calls.

Expected CSV columns: frame_index, x1, y1, x2, y2
(one row per detected ROI box; multiple rows can share the same
frame_index if there were multiple boxes in that frame)

Run from the REPO ROOT (not from inside src/track_det/):
    python -m src.track_det.run_on_p1_rois path/to/p1_rois.csv
"""
import sys
import csv
from collections import defaultdict

from .tracker import track, reset_tracker


def load_rois_by_frame(csv_path):
    """
    Returns: dict {frame_index: [(x1, y1, x2, y2), ...]}, sorted by
    frame_index ascending.
    """
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            box = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
            frames[frame_idx].append(box)
    return dict(sorted(frames.items()))


def main(csv_path):
    frames = load_rois_by_frame(csv_path)
    print(f"Loaded {len(frames)} frames from {csv_path}\n")

    for frame_idx, boxes in frames.items():
        tracks = track(boxes)
        ids = sorted(t["track_id"] for t in tracks)
        print(f"Frame {frame_idx}: {len(boxes)} input boxes -> "
              f"{len(tracks)} tracks, IDs={ids}")

    print("\nCheck: same person should keep the same ID across the clip; "
          "no ID jumping between people in crowded frames.")
    print("If IDs jump or churn a lot, first thing to tune: "
          "max_distance in CentroidTracker (src/track_det/centroid_tracker.py) "
          "against this clip's resolution/frame rate.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.track_det.run_on_p1_rois path/to/p1_rois.csv")
        sys.exit(1)
    main(sys.argv[1])