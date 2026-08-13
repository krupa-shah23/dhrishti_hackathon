"""
shake_compensation.py
Owner: P1 - Motion & ROI

Contract used by motion.py (do not change without team sign-off):
    FrameStabilizer.stabilize(frame) -> (stabilized_frame, (dx, dy))

Day-4 addition: global camera-shake compensation via phase correlation
(Option A — translation only, no rotation/perspective). Estimates
frame-to-frame translation caused by camera vibration and shifts each
frame back into alignment BEFORE it reaches MOG2, so vibration is not
misread as full-frame motion.

ORB-based rotation/perspective compensation (Option B) was considered
and rejected — unnecessary overhead for typical CCTV pole/mount vibration,
where translation dominates.

Pipeline position:
    Frame -> Stabilize (this module) -> MOG2 -> Morphology -> Contours -> ROIs
"""

import cv2
import numpy as np

# Tune if a clip shows over/under-correction; start with a conservative default.
DEFAULT_MAX_SHIFT = 50  # px — clamp on estimated translation per frame


class FrameStabilizer:
    """
    Keeps a running reference (previous) grayscale frame and estimates
    the (dx, dy) translation of each new frame relative to it via
    cv2.phaseCorrelate, then warps the new frame back into alignment.
    """

    def __init__(self, max_shift=DEFAULT_MAX_SHIFT, skip_threshold=4.0):
        """
        max_shift: clamp on estimated translation (pixels). Guards against
        phaseCorrelate producing a wild estimate on a low-texture frame
        (e.g. a mostly blank wall) — better to under-correct than to
        warp the frame into garbage.

        skip_threshold: if |dx| or |dy| exceeds this (pixels), skip the
        warp entirely and pass the original frame through unmodified.
        Empirically, per-frame phase-correlation warping measurably HELPS
        on mild jitter (~0.7px, e.g. boulevard/sidewalk: F1 +0.01 to +0.04)
        but measurably HURTS on severe jitter (~6-7px, e.g. badminton/
        traffic: F1 -0.01 to -0.05), since large single-frame translation
        estimates are noisier and the warp introduces more error than it
        removes. Gating avoids the failure mode without adding trajectory
        smoothing or other algorithmic complexity.
        """
        self.prev_gray = None
        self.max_shift = max_shift
        self.skip_threshold = skip_threshold

    def _to_gray_float(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.float32(gray)

    def stabilize(self, frame):
        """
        frame: BGR np.ndarray (single video frame)
        returns: (stabilized_frame, (dx, dy))
                 dx, dy = estimated camera shift for this frame (for logging/debug)
        First frame is always returned unchanged (nothing to compare against yet).
        """
        gray = self._to_gray_float(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame, (0.0, 0.0)

        # phaseCorrelate returns (shift, response). shift = (x, y) translation
        # of curr relative to prev.
        (dx, dy), _response = cv2.phaseCorrelate(self.prev_gray, gray)

        # Clamp to avoid wild warps on low-texture frames
        dx = float(np.clip(dx, -self.max_shift, self.max_shift))
        dy = float(np.clip(dy, -self.max_shift, self.max_shift))

        # Update reference to this frame's (unstabilized) grayscale so drift
        # is measured frame-to-frame, not against a single fixed frame.
        self.prev_gray = gray

        # Gate: severe single-frame shifts are noisier estimates and the
        # warp does more harm than good above this threshold (see F1 data
        # in __init__ docstring) — pass the frame through unmodified instead.
        if abs(dx) > self.skip_threshold or abs(dy) > self.skip_threshold:
            return frame, (dx, dy)

        # Shift the current frame back by (-dx, -dy) to cancel the camera motion
        h, w = frame.shape[:2]
        translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(
            frame, translation_matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        return stabilized, (dx, dy)

    def reset(self):
        """Call when starting a new clip so state doesn't leak across clips."""
        self.prev_gray = None