"""
crossing_swap_detector.py

Diagnostic tool: re-runs tracking on a p1_rois.csv file while recording
position history per track ID, then flags any (death, birth) pair that's
close in both time and space -- the signature of a crossing-induced ID
swap, as demonstrated synthetically in Master Doc SS5a.

This does NOT replace centroid_tracker.py -- it's a read-only analysis
script using the same documented parameters (Hungarian assignment,
max_distance=80.0, max_age=10) so results are directly comparable to
your shipped tracker's behavior.

Usage:
    uv run python -m src.track_det.crossing_swap_detector <csv_path> [--max-distance 80.0] [--max-age 10] [--swap-radius 80.0] [--swap-window 5]
"""
import sys
import argparse
import csv as csv_module
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_rois(csv_path):
    """frame_index -> list of (x1,y1,x2,y2) boxes"""
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frames[fi].append(box)
    return frames


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Track:
    __slots__ = ("id", "pos", "age", "first_frame", "last_frame", "last_pos", "birth_pos")

    def __init__(self, tid, pos, frame):
        self.id = tid
        self.pos = pos
        self.age = 0
        self.first_frame = frame
        self.last_frame = frame
        self.last_pos = pos
        self.birth_pos = pos


def run(csv_path, max_distance, max_age, swap_radius, swap_window):
    frames = load_rois(csv_path)
    if not frames:
        print("No ROI rows found in CSV.")
        return

    max_frame = max(frames.keys())
    next_id = 1
    active = {}          # id -> Track
    dead = []            # finished Track objects (aged out or run ended)

    for f in range(0, max_frame + 1):
        boxes = frames.get(f, [])
        centroids = [centroid(b) for b in boxes]

        active_ids = list(active.keys())
        matched_tracks = set()
        matched_det_indices = set()
        if active_ids and centroids:
            cost = np.zeros((len(active_ids), len(centroids)))
            for i, tid in enumerate(active_ids):
                for j, c in enumerate(centroids):
                    d = dist(active[tid].pos, c)
                    cost[i, j] = d if d <= max_distance else 1e6
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c_idx in zip(row_ind, col_ind):
                if cost[r, c_idx] <= max_distance:
                    tid = active_ids[r]
                    active[tid].pos = centroids[c_idx]
                    active[tid].age = 0
                    active[tid].last_frame = f
                    active[tid].last_pos = centroids[c_idx]
                    matched_tracks.add(tid)
                    matched_det_indices.add(c_idx)

        # age unmatched tracks, kill if too old
        for tid in list(active.keys()):
            if tid not in matched_tracks:
                active[tid].age += 1
                if active[tid].age > max_age:
                    dead.append(active.pop(tid))

        # spawn new tracks for unmatched detections
        for j, c in enumerate(centroids):
            if j not in matched_det_indices:
                t = Track(next_id, c, f)
                active[t.id] = t
                next_id += 1

    # flush remaining active tracks as "dead" at end of run
    dead.extend(active.values())
    dead.sort(key=lambda t: t.first_frame)

    print(f"--- Crossing-Swap Diagnostic: {csv_path} ---")
    print(f"Total frames: {max_frame + 1}, total tracks spawned: {next_id - 1}")
    print(f"Params: max_distance={max_distance}, max_age={max_age}, "
          f"swap_radius={swap_radius}, swap_window={swap_window} frames\n")

    flagged = []
    for death in dead:
        for birth in dead:
            if birth.id == death.id:
                continue
            gap = birth.first_frame - death.last_frame
            if 0 < gap <= swap_window:
                d = dist(death.last_pos, birth.birth_pos)
                if d <= swap_radius:
                    flagged.append((death.id, death.last_frame, death.last_pos,
                                     birth.id, birth.first_frame, birth.birth_pos, gap, d))

    if not flagged:
        print("No suspicious close death->birth pairs found within the given radius/window.")
    else:
        print(f"{len(flagged)} suspicious pair(s) found (possible crossing-swap signature):\n")
        for (did, dframe, dpos, bid, bframe, bpos, gap, d) in flagged:
            print(f"  Track {did} died @ frame {dframe} pos=({dpos[0]:.1f},{dpos[1]:.1f})  "
                  f"-> Track {bid} born @ frame {bframe} pos=({bpos[0]:.1f},{bpos[1]:.1f})  "
                  f"gap={gap} frames, dist={d:.1f}px")
        print("\nNote: this flags SPATIAL PROXIMITY only -- it does not confirm an actual swap, "
              "just candidates worth visually reviewing frame-by-frame (as in Master Doc SS5a).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--swap-radius", type=float, default=80.0,
                         help="max centroid distance between a death and a birth to flag as suspicious")
    parser.add_argument("--swap-window", type=int, default=5,
                         help="max frame gap between a death and a birth to consider")
    args = parser.parse_args()
    run(args.csv_path, args.max_distance, args.max_age, args.swap_radius, args.swap_window)
