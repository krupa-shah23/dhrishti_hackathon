"""
window_box_stats.py

Computes ROI box statistics (boxes-per-frame, box size, box count
variance) within specific frame windows -- used to diagnose whether
high track-ID fragmentation in a window is coming from noisy/fragmented
upstream ROI detection (P1's side) rather than tracker tuning.

Usage:
    uv run python -m src.track_det.window_box_stats \
        data/real_footage/subject1/subject1_rois.csv \
        --window 3750 5400 \
        --window 7625 7700
"""
import sys
import csv as csv_module
import argparse
from collections import defaultdict


def load_rois_by_frame(csv_path):
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frames[fi].append(box)
    return dict(frames)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def analyze_window(frames, start, end):
    total_frames = end - start + 1
    active_frames = 0
    box_counts = []
    areas = []

    for f in range(start, end + 1):
        boxes = frames.get(f, [])
        box_counts.append(len(boxes))
        if boxes:
            active_frames += 1
            for b in boxes:
                areas.append(box_area(b))

    avg_boxes_per_frame = sum(box_counts) / total_frames if total_frames else 0
    avg_boxes_per_active_frame = (sum(box_counts) / active_frames) if active_frames else 0
    max_boxes_in_a_frame = max(box_counts) if box_counts else 0

    print(f"  Window [{start}-{end}] ({total_frames} frames):")
    print(f"    Active frames: {active_frames}/{total_frames} "
          f"({100*active_frames/total_frames:.1f}%)")
    print(f"    Avg boxes/frame (all frames): {avg_boxes_per_frame:.3f}")
    print(f"    Avg boxes/frame (active frames only): {avg_boxes_per_active_frame:.3f}")
    print(f"    Max boxes in a single frame: {max_boxes_in_a_frame}")
    if areas:
        print(f"    Box area: min={min(areas):.0f}, max={max(areas):.0f}, "
              f"avg={sum(areas)/len(areas):.0f}")
        print(f"    Box area std/mean ratio (higher = more size variance): "
              f"{(max(areas)-min(areas))/(sum(areas)/len(areas)):.2f}")


def run(csv_path, windows):
    frames = load_rois_by_frame(csv_path)
    print(f"--- Window Box Stats: {csv_path} ---\n")
    for (start, end) in windows:
        analyze_window(frames, start, end)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--window", nargs=2, type=int, action="append",
                         metavar=("START", "END"), required=True,
                         help="Frame range to analyze; repeat --window for multiple ranges")
    args = parser.parse_args()
    run(args.csv_path, args.window)