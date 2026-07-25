"""
P2 module — Tracking.
Owns the shared function contract:
    track(boxes) -> [tracks]

Uses ByteTrack if installed/importable, else falls back automatically
to the centroid-distance tracker (no extra deps). Fallback choice is
logged once at import time so it's obvious in the console which one
is running.

RUNNING THIS MODULE / ANYTHING THAT IMPORTS IT:
Always run from the repo root using the -m flag, e.g.:
    python -m src.track_det.test_track
    python -m src.track_det.run_on_p1_rois path/to/p1_rois.csv
Do NOT `cd` into src/track_det/ and run scripts directly — the relative
imports below only resolve when Python sees this file as part of the
src.track_det package (which -m from repo root gives you).
"""
from typing import List, Tuple, Dict, Any

from .centroid_tracker_v2 import CentroidTracker

try:
    from .bytetrack_wrapper import ByteTrackWrapper, _HAS_BYTETRACK
except ImportError:
    _HAS_BYTETRACK = False

if _HAS_BYTETRACK:
    print("[tracker] Using ByteTrack.")
    _tracker = ByteTrackWrapper()
else:
    print("[tracker] ByteTrack not available — using CentroidTracker fallback.")
    _tracker = CentroidTracker()

def reset_tracker() -> None:
    """
    Resets the active tracker's internal state (all tracks dropped, ID
    counter restarted). Call this whenever you start processing a NEW
    video/clip in the same Python process — without it, track IDs and
    not-yet-aged-out tracks from the previous clip leak into the new
    one's results.

    Does NOT need to be called between frames of the SAME video — only
    between separate videos, or between isolated test cases.
    """
    global _tracker
    _tracker.reset()


def track(boxes: List[Tuple[float, float, float, float]]) -> List[Dict[str, Any]]:
    """
    Contract: track(boxes) -> [tracks]

    boxes: list of (x1, y1, x2, y2) for the CURRENT frame, from P1's
           get_rois(mask).
    returns: list of dicts, one per active track:
             {"track_id": int, "box": (x1, y1, x2, y2)}
    Call this once per frame, in frame order — the tracker keeps
    internal state (track ages, IDs) between calls.
    """
    if isinstance(_tracker, CentroidTracker):
        tracks = _tracker.update(boxes)
        return [{"track_id": t.track_id, "box": t.box} for t in tracks]
    else:
        results = _tracker.update(boxes)
        return [{"track_id": tid, "box": (x1, y1, x2, y2)}
                for tid, x1, y1, x2, y2 in results]