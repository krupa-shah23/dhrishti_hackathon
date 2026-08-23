"""
gt_overlap_check.py

Cross-references tracked activity against P1/OEP's gt.txt ground-truth
behavior windows. Two checks:
  1. Coverage: does ROI motion actually occur during labeled windows?
  2. Fragmentation-per-window: how many distinct track_ids appear during
     each labeled window? (A cleaner metric than whole-clip MaxID, since
     it's scoped to a real, meaningful time interval.)

gt.txt format assumed: "MMSS  MMSS  label_id" per line (start, end, label),
converted to frame numbers using the clip's real FPS.

Usage:
    uv run python -m src.track_det.gt_overlap_check \
        data/real_footage/subject1/subject1_rois.csv \
        data/real_footage/subject1/gt.txt \
        --fps 25 --max-distance 80.0 --max-age 10
"""
import sys
import csv as csv_module
import argparse
from collections import defaultdict

from .tracker import track, reset_tracker


def mmss_to_seconds(mmss_str):
    """'1339' -> 13 min 39 sec -> 819 seconds"""
    mmss_str = mmss_str.strip().zfill(4)
    minutes = int(mmss_str[:-2])
    seconds = int(mmss_str[-2:])
    return minutes * 60 + seconds


def load_gt(gt_path, fps):
    windows = []  # (start_frame, end_frame, label)
    with open(gt_path, newline="") as f:
        reader = csv_module.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                # try whitespace-split fallback if not tab-delimited
                row = row[0].split() if row else []
                if len(row) < 3:
                    continue
            start_s = mmss_to_seconds(row[0])
            end_s = mmss_to_seconds(row[1])
            label = row[2].strip()
            start_frame = int(start_s * fps)
            end_frame = int(end_s * fps)
            windows.append((start_frame, end_frame, label))
    return windows


def load_rois_by_frame(csv_path):
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frames[fi].append(box)
    return dict(frames)


def run(csv_path, gt_path, fps, max_distance, max_age):
    windows = load_gt(gt_path, fps)
    frames = load_rois_by_frame(csv_path)
    max_frame = max(frames.keys())

    reset_tracker(max_distance=max_distance, max_age=max_age)

    # Walk every real frame, track it, and record which track_ids were
    # active on each frame -- needed to compute per-window fragmentation.
    frame_track_ids = {}  # frame_idx -> set of track_ids active that frame
    for f in range(0, max_frame + 1):
        boxes = frames.get(f, [])
        current = track(boxes)
        frame_track_ids[f] = set(t["track_id"] for t in current)

    print(f"--- GT Overlap Check: {csv_path} vs {gt_path} ---")
    print(f"Params: fps={fps}, max_distance={max_distance}, max_age={max_age}")
    print(f"Total real frames: {max_frame + 1}, labeled windows: {len(windows)}\n")

    covered = 0
    uncovered = []
    label_track_counts = defaultdict(list)  # label -> list of distinct-ID-counts per window

    for (start, end, label) in windows:
        window_track_ids = set()
        has_motion = False
        for f in range(start, min(end, max_frame) + 1):
            if frames.get(f):
                has_motion = True
            window_track_ids.update(frame_track_ids.get(f, set()))

        if has_motion:
            covered += 1
        else:
            uncovered.append((start, end, label))

        label_track_counts[label].append(len(window_track_ids))

        print(f"  Window [{start}-{end}] label={label}: "
              f"motion={'YES' if has_motion else 'NO'}, "
              f"distinct track_ids in window={len(window_track_ids)}")

    print(f"\n--- Summary ---")
    print(f"Windows with detected motion overlap: {covered}/{len(windows)} "
          f"({100*covered/len(windows):.1f}%)")
    if uncovered:
        print(f"Windows with ZERO motion overlap (worth investigating):")
        for (s, e, l) in uncovered:
            print(f"  [{s}-{e}] label={l}")

    print(f"\nAvg distinct track_ids per window, by label:")
    for label, counts in sorted(label_track_counts.items()):
        avg = sum(counts) / len(counts)
        print(f"  label={label}: {len(counts)} windows, avg {avg:.2f} distinct IDs, "
              f"range {min(counts)}-{max(counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("gt_path")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-age", type=int, default=10)
    args = parser.parse_args()
    run(args.csv_path, args.gt_path, args.fps, args.max_distance, args.max_age)