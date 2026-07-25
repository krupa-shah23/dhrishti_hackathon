"""
Runs the tracker across ALL 9 real CDNet2014 CSVs P1 provided, producing
a single comparison table -- this is the cross-dataset generalization
evidence flagged as a strong differentiator in the team's own brainstorm
doc: "run the same unmodified pipeline on several genuinely different
videos, show it performs reasonably without per-video retuning."

IMPORTANT CONTEXT: these are CDNet2014 benchmark/dev clips (per the
team's own dataset plan), NOT final exam-hall footage, and CDNet stores
raw image sequences with NO embedded FPS. Gap lengths below are reported
in FRAMES only, not seconds/real-time.

TUNING NOTE: two per-clip max_distance tuning attempts (jitter-based,
and real-measured-displacement-based) were tried and BOTH increased
fragmentation on every one of the 9 clips relative to the uniform 80.0
default -- see team notes for the full before/after comparison and why
(MaxID alone can't validate max_distance without ground-truth identity).
This script has been reverted to the uniform max_distance=80.0 default.

MATCHING NOTE: centroid_tracker.py now uses Hungarian assignment
(scipy.optimize.linear_sum_assignment) instead of greedy nearest-centroid
matching. This run isolates that as the ONLY change from the original
locked baseline (badminton=666, cubicle=111, etc.) -- max_distance is
back to the same uniform 80.0 used in that baseline, so any difference
in the numbers below is attributable to the matching algorithm change,
not a threshold change.

Run from the REPO ROOT:
    python -m src.track_det.batch_compare_rois

Expects all p1_rois_<name>.csv files to be at src/p1_rois_<name>.csv.
"""
import csv
from collections import defaultdict

from .tracker import track, reset_tracker

MAX_AGE_REFERENCE = 10  # must match CentroidTracker's default max_age
DEFAULT_MAX_DISTANCE = 80.0  # the original, twice-verified default

SEQUENCES = [
    ("p1_rois_badminton.csv",         "badminton (cameraJitter)",       720, 480),
    ("p1_rois_boulevard.csv",         "boulevard (cameraJitter)",       352, 240),
    ("p1_rois_sidewalk.csv",          "sidewalk (cameraJitter)",        352, 240),
    ("p1_rois_traffic.csv",           "traffic (cameraJitter)",         320, 240),
    ("p1_rois_fountain02.csv",        "fountain02 (dynamicBackground)", 432, 288),
    ("p1_rois_tramCrossroad_1fps.csv","tramCrossroad_1fps (lowFramerate)", 640, 350),
    ("p1_rois_backdoor.csv",          "backdoor (shadow)",              320, 240),
    ("p1_rois_copyMachine.csv",       "copyMachine (shadow)",           720, 480),
    ("p1_rois_cubicle.csv",           "cubicle (shadow)",               352, 240),
]


def load_rois_by_frame(csv_path):
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            box = (
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            )
            frames[frame_idx].append(box)
    return dict(frames)


def analyze_sequence(csv_path, width, height):
    frames = load_rois_by_frame(csv_path)
    if not frames:
        return None

    max_frame = max(frames.keys())
    reset_tracker(max_distance=DEFAULT_MAX_DISTANCE)  # explicit, uniform, matches baseline

    max_track_id_seen = 0
    gap_events = []
    current_gap_start = None

    for frame_idx in range(max_frame + 1):
        boxes = frames.get(frame_idx, [])
        tracks = track(boxes)

        if boxes:
            if current_gap_start is not None:
                gap_len = frame_idx - current_gap_start
                if gap_len > MAX_AGE_REFERENCE:
                    gap_events.append(gap_len)
                current_gap_start = None
        elif current_gap_start is None:
            current_gap_start = frame_idx

        for t in tracks:
            max_track_id_seen = max(max_track_id_seen, t["track_id"])

    return {
        "total_frames": max_frame + 1,
        "active_frames": len(frames),
        "active_pct": round(100 * len(frames) / (max_frame + 1), 1),
        "max_track_id": max_track_id_seen,
        "num_gaps_over_max_age": len(gap_events),
        "longest_gap": max(gap_events) if gap_events else 0,
    }


def main():
    print("Cross-dataset generalization check -- HUNGARIAN assignment, "
          f"uniform max_distance={DEFAULT_MAX_DISTANCE} (same as original "
          "baseline), run across 9 real, genuinely different clips.\n")
    print("NOTE: gap lengths are in FRAMES, not seconds -- these clips have "
          "no embedded FPS (CDNet2014 image-sequence format).\n")

    results = []
    for filename, label, width, height in SEQUENCES:
        path = f"src/{filename}"
        try:
            stats = analyze_sequence(path, width, height)
        except FileNotFoundError:
            print(f"SKIPPED (file not found): {path}")
            continue
        if stats is None:
            print(f"SKIPPED (no ROI rows loaded): {path}")
            continue
        results.append((label, width, height, stats))

    if not results:
        print("No files found.")
        return

    header = (f"{'Sequence':<32} {'Res':<10} {'Active%':>8} {'MaxID':>6} "
              f"{'Gaps>10':>8} {'LongestGap':>11}")
    print(header)
    print("-" * len(header))
    for label, width, height, s in results:
        print(f"{label:<32} {width}x{height:<6} {s['active_pct']:>7.1f}% "
              f"{s['max_track_id']:>6} {s['num_gaps_over_max_age']:>8} "
              f"{s['longest_gap']:>11}")

    print(
        "\nMaxID: highest track ID reached, using HUNGARIAN assignment "
        "at uniform max_distance=80.0. Compare directly against the "
        "original greedy-matching baseline (badminton=666, boulevard=136, "
        "sidewalk=145, traffic=230, fountain02=36, tramCrossroad=339, "
        "backdoor=50, copyMachine=316, cubicle=111) to isolate the effect "
        "of the matching algorithm alone."
    )


if __name__ == "__main__":
    main()