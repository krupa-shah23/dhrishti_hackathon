"""
Centroid-distance tracker with Hungarian (optimal) assignment.

UPDATED: matching changed from greedy nearest-centroid (process tracks
one at a time, each picking its own nearest unclaimed box) to Hungarian
assignment via scipy.optimize.linear_sum_assignment -- this finds the
GLOBALLY minimum-total-distance matching across the whole frame at once,
rather than committing to locally-good matches that can block a better
global configuration.

Note: this does NOT fix the perfectly symmetric crossing-swap case
already documented (see centroid_tracker_v2.py's docstring and the
master doc §5a) -- at a truly symmetric crossing, the swapped pairing
IS the minimum-total-distance solution, so Hungarian assignment will
reproduce it too. What Hungarian DOES fix: the many non-symmetric,
near-miss cases where greedy's "whichever track has the lowest ID
picks first" ordering was making a locally-good-but-globally-suboptimal
choice. Real crossings are rarely perfectly symmetric, so this should
measurably help on most of the 9-clip dataset.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    age: int = 0
    hits: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_age: int = 10):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes: List[Tuple[float, float, float, float]]) -> List[Track]:
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_boxes = len(boxes)

        matched_track_ids = set()
        matched_box_indices = set()

        if n_tracks > 0 and n_boxes > 0:
            # Build a full cost matrix (tracks x boxes). Pairs beyond
            # max_distance get a very high cost so Hungarian naturally
            # avoids assigning them -- same effect as the old "only
            # consider pairs within max_distance" filter, but expressed
            # as a cost instead of a hard exclusion (linear_sum_assignment
            # needs a complete matrix, not a sparse candidate list).
            INFEASIBLE_COST = 1e9
            cost_matrix = np.full((n_tracks, n_boxes), INFEASIBLE_COST)

            track_centroids = [_centroid(self.tracks[tid].box) for tid in track_ids]
            box_centroids = [_centroid(b) for b in boxes]

            for i, tc in enumerate(track_centroids):
                for j, bc in enumerate(box_centroids):
                    d = _dist(tc, bc)
                    if d <= self.max_distance:
                        cost_matrix[i, j] = d

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] >= INFEASIBLE_COST:
                    continue  # this "match" was only assigned because
                              # linear_sum_assignment forces a complete
                              # matching -- it's not a real match, skip it
                tid = track_ids[row]
                track = self.tracks[tid]
                track.box = boxes[col]
                track.age = 0
                track.hits += 1
                track.history.append(boxes[col])
                matched_track_ids.add(tid)
                matched_box_indices.add(col)

        # age out unmatched tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                track = self.tracks[tid]
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[tid]

        # spawn new tracks for leftover boxes
        for bi in range(n_boxes):
            if bi not in matched_box_indices:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(track_id=tid, box=boxes[bi], age=0, hits=1,
                                          history=[boxes[bi]])

        return list(self.tracks.values())