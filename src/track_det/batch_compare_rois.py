"""
Runs the (fixed, time-correct) tracker across ALL 9 real CDNet2014 CSVs
P1 provided, producing a single comparison table -- this is the
cross-dataset generalization evidence flagged as a strong differentiator
in the team's own brainstorm doc: "run the same unmodified pipeline on
several genuinely different videos, show it performs reasonably
without per-video retuning."

IMPORTANT CONTEXT: these are CDNet2014 benchmark/dev clips (per the
team's own dataset plan), NOT final exam-hall footage, and CDNet stores
raw image sequences with NO embedded FPS. That means:
- Gap lengths below are reported in FRAMES only, not seconds/real-time
  -- do not convert to seconds without real FPS, which doesn't exist
  for this dataset.
- max_distance recommendations DO scale meaningfully by resolution
  (geometry doesn't need FPS) -- these are included per-sequence.
- max_age recommendations do NOT attempt a real-time conversion here,
  since there's no FPS to convert with. Treat max_age tuning as a
  separate exercise once real exam-hall footage (with real FPS)
  arrives.

Run from the REPO ROOT:
    python -m src.track_det.batch_compare_rois

Expects all p1_rois_<name>.csv files to be at src/p1_rois_<name>.csv
(same location P1 has them in her branch). Edit SEQUENCES below if
your local paths differ.
"""
import csv
from collections import defaultdict

from .tracker import track, reset_tracker

MAX_AGE_REFERENCE = 10  # must match CentroidTracker's default max_age

# (csv_filename, display_name, width, height) -- resolution from P1's
# video_info.py output, used only to compute a per-sequence max_distance
# suggestion (jitter scales with resolution; this does NOT need FPS).
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

REFERENCE_HEIGHT = 240   # height of the clip the ~5px jitter reference was measured on
OBSERVED_JITTER_PX = 5.0
SAFETY_MULTIPLIER = 6.0


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
    reset_tracker()

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

    scale_factor = height / REFERENCE_HEIGHT
    suggested_max_distance = round(OBSERVED_JITTER_PX * scale_factor * SAFETY_MULTIPLIER, 1)

    return {
        "total_frames": max_frame + 1,
        "active_frames": len(frames),
        "active_pct": round(100 * len(frames) / (max_frame + 1), 1),
        "max_track_id": max_track_id_seen,
        "num_gaps_over_max_age": len(gap_events),
        "longest_gap": max(gap_events) if gap_events else 0,
        "suggested_max_distance": suggested_max_distance,
    }


def main():
    print("Cross-dataset generalization check -- same unmodified tracker, "
          f"same default settings (max_distance=80.0, max_age={MAX_AGE_REFERENCE}), "
          "run across 9 real, genuinely different clips.\n")
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
        print("No files found. Make sure the CSVs are at src/p1_rois_<name>.csv "
              "(same layout as P1's branch).")
        return

    header = (f"{'Sequence':<32} {'Res':<10} {'Active%':>8} {'MaxID':>6} "
              f"{'Gaps>10':>8} {'LongestGap':>11} {'SuggMaxDist':>12}")
    print(header)
    print("-" * len(header))
    for label, width, height, s in results:
        print(f"{label:<32} {width}x{height:<6} {s['active_pct']:>7.1f}% "
              f"{s['max_track_id']:>6} {s['num_gaps_over_max_age']:>8} "
              f"{s['longest_gap']:>11} {s['suggested_max_distance']:>12}")

    print(
        "\nActive%: fraction of total frames that had at least one ROI "
        "(low active% = mostly static scene, e.g. sparse motion).\n"
        "MaxID: highest track ID reached -- a rough proxy for how "
        "fragmented tracking was on this sequence under CURRENT default "
        "settings (unmodified across all 9 -- this IS the generalization "
        "test).\n"
        "SuggMaxDistance: a per-resolution starting-point suggestion "
        "(scaled from real measured jitter), NOT yet what the tracker "
        "actually used in this run -- this run intentionally used the "
        "SAME default (80.0) everywhere to test generalization first."
    )


if __name__ == "__main__":
    main()