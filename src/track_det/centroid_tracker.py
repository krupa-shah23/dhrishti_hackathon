"""
Centroid-distance tracker — fallback for ByteTrack.
No extra dependencies. Matches boxes frame-to-frame by nearest centroid.

Good enough for a first pass / low-crowd-density scenes. Swap for
ByteTrack (bytetrack_wrapper.py) once cython-bbox/lap install cleanly
and you need robust ID continuity in dense/occluded scenes.
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
        unmatched_boxes = list(range(len(boxes)))
        matched_track_ids = set()

        # greedy nearest-centroid matching, existing tracks first
        for tid, track in list(self.tracks.items()):
            if not unmatched_boxes:
                break
            track_c = _centroid(track.box)
            best_idx, best_dist = None, None
            for bi in unmatched_boxes:
                d = _dist(track_c, _centroid(boxes[bi]))
                if best_dist is None or d < best_dist:
                    best_idx, best_dist = bi, d
            if best_dist is not None and best_dist <= self.max_distance:
                track.box = boxes[best_idx]
                track.age = 0
                track.hits += 1
                track.history.append(boxes[best_idx])
                matched_track_ids.add(tid)
                unmatched_boxes.remove(best_idx)

        # age out unmatched tracks
        for tid, track in list(self.tracks.items()):
            if tid not in matched_track_ids:
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[tid]

        # spawn new tracks for leftover boxes
        for bi in unmatched_boxes:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, box=boxes[bi], age=0, hits=1,
                                      history=[boxes[bi]])

        return list(self.tracks.values())