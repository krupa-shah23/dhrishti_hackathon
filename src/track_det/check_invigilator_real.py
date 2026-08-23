"""
check_invigilator_real.py

Sanity check (NOT validation) of is_invigilator_track() against real
footage. No ground-truth invigilator/student label exists in 01_001 or
01_0015 -- this cannot confirm the heuristic is correct, only that it
runs cleanly on real data and produces plausible, inspectable numbers.

Approach: replay the real p1_rois CSV through CentroidTracker exactly
like run_on_p1_rois.py does (frame-accurate, calling track([]) for
empty frames so max_age semantics stay correct -- see Master Doc §7a),
but snapshot each track's full history the moment it dies (ages out),
since CentroidTracker deletes dead tracks from its internal dict and
their history would otherwise be lost.

Usage:
    uv run python -m src.track_det.check_invigilator_real data/real_footage/01_001/p1_rois_01_001.csv
    uv run python -m src.track_det.check_invigilator_real data/real_footage/01_0015/p1_rois_01_0015.csv
"""
import argparse
import csv
from collections import defaultdict

from .centroid_tracker import CentroidTracker
from .invigilator_filter import is_invigilator_track, path_length, net_displacement


def load_rois(csv_path):
    """
    frame_index,x1,y1,x2,y2 -- one row per ROI, frames with zero ROIs
    omitted entirely (per P1's handoff format, Master Doc §7).
    """
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frames[fi].append(box)
    return frames


def run(csv_path, min_hits=15, min_net_displacement=150.0, min_path_length=150.0):
    frames = load_rois(csv_path)
    max_frame = max(frames.keys()) if frames else -1

    tracker = CentroidTracker(max_distance=80.0, max_age=10)
    dead_track_histories = {}  # track_id -> history, captured at death

    prev_ids = set()
    for fi in range(max_frame + 1):
        boxes = frames.get(fi, [])
        tracks = tracker.update(boxes)
        current_ids = {t.track_id for t in tracks}

        # any track present last frame but gone now just aged out --
        # capture its history before it's lost for good. We rebuild it
        # from what we can see: since update() doesn't return dead
        # tracks, we snapshot histories of all CURRENTLY LIVE tracks
        # every frame, so whatever was last seen right before death is
        # preserved.
        for t in tracks:
            dead_track_histories[t.track_id] = list(t.history)

        prev_ids = current_ids

    print(f"\n--- Invigilator-filter sanity check: {csv_path} ---")
    print(f"Total frames: {max_frame + 1}")
    print(f"Total tracks ever created: {tracker.next_id - 1}")
    print(f"Thresholds: min_hits={min_hits}, "
          f"min_net_displacement={min_net_displacement}, "
          f"min_path_length={min_path_length}\n")

    flagged = []
    not_flagged_but_close = []

    for tid in sorted(dead_track_histories.keys()):
        hist = dead_track_histories[tid]
        if len(hist) < 2:
            continue  # not enough data to compute anything meaningful

        pl = path_length(hist)
        nd = net_displacement(hist)
        hits = len(hist)
        flag = is_invigilator_track(
            hist,
            min_hits=min_hits,
            min_net_displacement=min_net_displacement,
            min_path_length=min_path_length,
        )

        print(f"Track {tid:4d}: hits={hits:4d}  path_length={pl:8.1f}  "
              f"net_displacement={nd:8.1f}  -> {'FLAGGED' if flag else 'not flagged'}")

        if flag:
            flagged.append(tid)
        elif hits >= min_hits and (pl >= min_path_length * 0.7 or nd >= min_net_displacement * 0.7):
            # close to threshold but didn't cross -- worth a human glance
            not_flagged_but_close.append(tid)

    print(f"\nSummary: {len(flagged)} track(s) flagged as invigilator-like "
          f"out of {len(dead_track_histories)} tracks with any history.")
    if flagged:
        print(f"Flagged track IDs: {flagged}")
    if not_flagged_but_close:
        print(f"Tracks close to threshold but NOT flagged (worth a manual "
              f"look): {not_flagged_but_close}")

    print("\nReminder: this is a SANITY CHECK, not a validation -- there is "
          "no ground-truth invigilator/student label in this footage. A "
          "flagged track here is not confirmed correct or incorrect without "
          "a human watching the corresponding real video segment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--min-hits", type=int, default=15)
    parser.add_argument("--min-net-displacement", type=float, default=150.0)
    parser.add_argument("--min-path-length", type=float, default=150.0)
    args = parser.parse_args()
    run(args.csv_path, args.min_hits, args.min_net_displacement, args.min_path_length)