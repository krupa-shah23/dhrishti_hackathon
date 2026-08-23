"""
ByteTrack wrapper. Uses the `bytetracker` pip package if it's installed
and importable; otherwise raises ImportError so track_det.py can fall
back to CentroidTracker automatically.

If `bytetracker` isn't on PyPI in a usable state for you, vendor the
official repo (https://github.com/ifzhang/ByteTrack) into this folder
and adjust the import below.
"""
from typing import List, Tuple
import numpy as np

try:
    from bytetracker import BYTETracker  # pip package
    _HAS_BYTETRACK = True
except ImportError:
    _HAS_BYTETRACK = False


class ByteTrackWrapper:
    def __init__(self, frame_rate: int = 25, track_thresh: float = 0.5,
                 match_thresh: float = 0.8, track_buffer: int = 30):
        if not _HAS_BYTETRACK:
            raise ImportError(
                "bytetracker not installed/importable — "
                "use CentroidTracker fallback instead."
            )
        self.tracker = BYTETracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
        )

    def reset(self):
        """
        Re-initializes the underlying BYTETracker instance, discarding
        all track state. Same purpose as CentroidTracker.reset() — call
        between separate videos/clips.
        """
        self.tracker = BYTETracker(
            frame_rate=self.tracker.frame_rate,
            track_thresh=self.tracker.track_thresh,
            match_thresh=self.tracker.match_thresh,
            track_buffer=self.tracker.track_buffer,
        )

    def update(self, boxes: List[Tuple[float, float, float, float]],
               scores: List[float] = None):
        """
        boxes: list of (x1, y1, x2, y2) from get_rois(mask)
        scores: optional confidence per box (defaults to 1.0 — MOG2/contour
                boxes don't have a natural confidence score, unlike a
                detector's output)
        returns: list of (track_id, x1, y1, x2, y2)
        """
        if scores is None:
            scores = [1.0] * len(boxes)
        dets = np.array(
            [[x1, y1, x2, y2, s] for (x1, y1, x2, y2), s in zip(boxes, scores)],
            dtype=np.float32,
        ) if boxes else np.empty((0, 5), dtype=np.float32)

        online_targets = self.tracker.update(dets)
        results = []
        for t in online_targets:
            x1, y1, w, h, tid = t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3], t.track_id
            results.append((int(tid), x1, y1, x1 + w, y1 + h))
        return results