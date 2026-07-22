"""
motion.py
Owner: P1 - Motion & ROI

Shared contract (do not change signature without team sign-off):
    get_motion_mask(frame) -> mask
"""

import cv2
import numpy as np


class MotionEstimator:
    """
    Wraps MOG2 background subtraction so we can keep per-clip state
    (the subtractor needs to see frames in sequence to build its model).
    """

    def __init__(self, history=500, var_threshold=25, detect_shadows=True, learning_rate=0.0008):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.learning_rate = learning_rate  # explicit, slower than MOG2's default 1/history ≈ 0.002, so near-stationary subjects survive longer before being absorbed into the background model

    def get_motion_mask(self, frame):
        """
        frame: BGR np.ndarray (single video frame)
        returns: binary mask (np.uint8, 0/255), shadows already excluded
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        raw_mask = self.mog2.apply(blurred, learningRate=self.learning_rate)

        # MOG2 with detectShadows=True marks shadow pixels as 127 (gray).
        # We only want true foreground (255), so threshold shadows out.
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        return binary_mask


def motion_intensity(mask):
    """
    Quick per-frame motion score: fraction of pixels flagged as foreground.
    Useful for the motion-intensity-per-frame array + later timeline plotting.
    """
    return float(np.count_nonzero(mask)) / mask.size


def process_clip(folder_path, history=500, var_threshold=16):
    """
    Runs MOG2 over a CDNet-style image-sequence clip (folder of numbered frames,
    e.g. input/in000001.jpg ...), since CDNet provides sequences, not video files.
    Returns: list of masks (np.ndarray), list of per-frame motion intensity floats
    """
    import os

    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    )
    if not files:
        raise FileNotFoundError(f"No frame images found in: {folder_path}")

    estimator = MotionEstimator(history=history, var_threshold=var_threshold)
    masks = []
    intensities = []

    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is None:
            continue
        mask = estimator.get_motion_mask(frame)
        masks.append(mask)
        intensities.append(motion_intensity(mask))

    return masks, intensities

# Module-level convenience function matching the exact shared contract
# signature: get_motion_mask(frame) -> mask
# For single-frame calls outside a stateful loop (e.g. quick tests),
# spin up a fresh estimator. In the real pipeline, prefer MotionEstimator
# directly so background modeling persists across frames.
_default_estimator = None


def get_motion_mask(frame):
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = MotionEstimator()
    return _default_estimator.get_motion_mask(frame)