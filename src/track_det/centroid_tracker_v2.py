# UNUSED — EXPERIMENTAL, DO NOT IMPORT. See docstring below for why.
"""
Centroid-distance tracker — fallback for ByteTrack.
No extra dependencies. Matches boxes frame-to-frame by nearest centroid.

UPDATED THIS SESSION: matching changed from per-track greedy (process
existing tracks one at a time, in dict-iteration/ID order, each picking
its own nearest box) to GLOBAL minimum-distance matching (compute every
valid track<->box distance pair up front, sort smallest-first, assign
greedily from that global ordering). This is still not true Hungarian
assignment (which optimizes total assignment cost, not just greedy
nearest-first), but it fixes the specific failure mode demonstrated in
test_track_stress.py's crossing-paths case: with the OLD per-track
loop, whichever track happened to be processed first (lowest ID) got
first pick of the nearest box, which could be the WRONG box right at
the moment two people's paths cross. With global sorting, the single
closest pair in the whole frame gets matched first, regardless of
which track ID it belongs to -- this is the harder-to-fool ordering.

KNOWN REMAINING LIMITATION: this still is not full Hungarian
assignment, so it does not guarantee the GLOBALLY OPTIMAL total-
distance matching in every configuration -- only that ties are broken
by "closest pair first" rather than "lowest track ID first". Dense,
highly ambiguous crossings (3+ people converging at once) can still
mismatch. Documented as an improvement, not a complete fix.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    age: int = 0            # frames since last matched
    hits: int = 0           # total times matched
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_age: int = 10):
        """
        max_distance: max centroid movement (px) allowed to still count as
                      the same track between consecutive frames. Tune this
                      to your frame size / subject speed.
        max_age:      frames a track can go unmatched before being dropped
                      (handles brief occlusion).
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def reset(self):
        """
        Clears all active tracks and resets ID counter back to 1.
        Call this between separate videos/clips — without it, IDs and
        phantom (aged-but-not-yet-dropped) tracks from a previous clip
        leak into the next one, corrupting results.
        """
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes: List[Tuple[float, float, float, float]]) -> List[Track]:
        """
        boxes: list of (x1, y1, x2, y2) for the current frame, from
               get_rois(mask).
        returns: list of active Track objects with persistent track_id.
        """
        track_ids = list(self.tracks.keys())
        box_indices = list(range(len(boxes)))

        # Build every valid (track, box) pair within max_distance, with
        # its distance -- valid meaning "close enough to even consider",
        # not yet "assigned".
        candidate_pairs = []  # (distance, track_id, box_idx)
        for tid in track_ids:
            track_c = _centroid(self.tracks[tid].box)
            for bi in box_indices:
                d = _dist(track_c, _centroid(boxes[bi]))
                if d <= self.max_distance:
                    candidate_pairs.append((d, tid, bi))

        # GLOBAL greedy: sort ALL candidate pairs by distance ascending,
        # then walk the list assigning the closest pair first -- once a
        # track or box is used, skip any later (worse) pair involving it.
        # This is the key change from the old per-track loop: matching
        # order is now determined by "which pair is objectively closest
        # across the WHOLE frame", not "which track has the lowest ID".
        candidate_pairs.sort(key=lambda p: p[0])

        matched_track_ids = set()
        matched_box_indices = set()
        for dist, tid, bi in candidate_pairs:
            if tid in matched_track_ids or bi in matched_box_indices:
                continue
            track = self.tracks[tid]
            track.box = boxes[bi]
            track.age = 0
            track.hits += 1
            track.history.append(boxes[bi])
            matched_track_ids.add(tid)
            matched_box_indices.add(bi)

        # age out unmatched tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                track = self.tracks[tid]
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[tid]

        # spawn new tracks for leftover boxes
        for bi in box_indices:
            if bi not in matched_box_indices:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(track_id=tid, box=boxes[bi], age=0, hits=1,
                                          history=[boxes[bi]])

        return list(self.tracks.values())