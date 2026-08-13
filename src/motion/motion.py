"""
motion.py
Owner: P1 - Motion & ROI

Shared contract (do not change signature without team sign-off):
    get_motion_mask(frame) -> mask
"""

import cv2
import numpy as np

from shake_compensation import FrameStabilizer


class MotionEstimator:
    """
    Wraps MOG2 background subtraction so we can keep per-clip state
    (the subtractor needs to see frames in sequence to build its model).
    """

    def __init__(self, history=500, var_threshold=25, detect_shadows=True,
                 learning_rate=0.0008, use_stabilization=False):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.learning_rate = learning_rate  # explicit, slower than MOG2's default 1/history ≈ 0.002, so near-stationary subjects survive longer before being absorbed into the background model

        self.use_stabilization = use_stabilization
        self.stabilizer = FrameStabilizer() if use_stabilization else None
        self.last_shift = (0.0, 0.0)  # last estimated (dx, dy), for logging/comparison

        # Invigilator / ignore-region hook: another module can populate this
        # with [(x1, y1, x2, y2), ...] boxes; motion in these regions is
        # zeroed out of the mask before it ever reaches get_rois(). This
        # module never detects the invigilator itself — it only provides
        # the hook. Distinct from roi.py's exclusion_regions, which are
        # static permanent-background rectangles validated per clip; this
        # is for a dynamic region another module updates at runtime.
        self.ignore_regions = []

    def set_ignore_regions(self, regions):
        """
        regions: [(x1, y1, x2, y2), ...] in frame pixel coordinates.
        """
        self.ignore_regions = regions

    def get_motion_mask(self, frame):
        """
        frame: BGR np.ndarray (single video frame)
        returns: binary mask (np.uint8, 0/255), shadows excluded,
                 ignore_regions zeroed out, camera-shake compensated if enabled
        """
        if self.use_stabilization:
            frame, self.last_shift = self.stabilizer.stabilize(frame)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        raw_mask = self.mog2.apply(blurred, learningRate=self.learning_rate)

        # MOG2 with detectShadows=True marks shadow pixels as 127 (gray).
        # We only want true foreground (255), so threshold shadows out.
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        for (x1, y1, x2, y2) in self.ignore_regions:
            binary_mask[y1:y2, x1:x2] = 0

        return binary_mask

    def reset(self):
        """Call when starting a new clip so stabilizer state doesn't leak across clips."""
        if self.stabilizer is not None:
            self.stabilizer.reset()


def motion_intensity(mask):
    """
    Quick per-frame motion score: fraction of pixels flagged as foreground.
    Useful for the motion-intensity-per-frame array + later timeline plotting.
    """
    return float(np.count_nonzero(mask)) / mask.size


def process_clip(folder_path, history=500, var_threshold=25,
                  learning_rate=0.0008, use_stabilization=False):
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

    estimator = MotionEstimator(history=history, var_threshold=var_threshold,
                                 learning_rate=learning_rate,
                                 use_stabilization=use_stabilization)
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

def process_video(video_path, history=500, var_threshold=25,
                   learning_rate=0.0008, use_stabilization=False):
    """
    Runs MOG2 over a single video file (e.g. MSU OEP .avi clips) — first
    genuinely video-file dataset in the pipeline; CDNet and ShanghaiTech
    are both image sequences and use process_clip() instead.
    Returns: list of masks, list of per-frame motion intensity floats,
             and the source video's real (measured, not assumed) fps/resolution/frame_count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator = MotionEstimator(history=history, var_threshold=var_threshold,
                                 learning_rate=learning_rate,
                                 use_stabilization=use_stabilization)
    masks = []
    intensities = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = estimator.get_motion_mask(frame)
        masks.append(mask)
        intensities.append(motion_intensity(mask))

    cap.release()

    metadata = {"fps": fps, "width": width, "height": height, "frame_count": frame_count}
    return masks, intensities, metadata

def get_motion_mask(frame):
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = MotionEstimator()
    return _default_estimator.get_motion_mask(frame)