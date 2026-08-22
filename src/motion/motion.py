"""
motion.py
Owner: P1 - Motion & ROI

Shared contract (do not change signature without team sign-off):
    get_motion_mask(frame) -> mask
"""

import cv2
import numpy as np

try:
    from .shake_compensation import FrameStabilizer
except ImportError:
    from shake_compensation import FrameStabilizer



class MotionEstimator:
    """
    Wraps MOG2 background subtraction so we can keep per-clip state
    (the subtractor needs to see frames in sequence to build its model).
    """

    def __init__(self, history=500, var_threshold=25, detect_shadows=True,
                 learning_rate=0.0008, use_stabilization=False, camera_mask=None):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.learning_rate = learning_rate  # explicit, slower than MOG2's default 1/history ≈ 0.002, so near-stationary subjects survive longer before being absorbed into the background model

        self.use_stabilization = use_stabilization
        self.stabilizer = FrameStabilizer() if use_stabilization else None
        self.last_shift = (0.0, 0.0)  # last estimated (dx, dy), for logging/comparison
        
        self.camera_mask = camera_mask

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

    def get_motion_mask(self, frame, mask=None):
        """
        frame: BGR np.ndarray (single video frame)
        mask: optional binary exclusion mask (0 to ignore, 255 to process)
        returns: (mag_map, mog2_mask) where mag_map is absolute frame difference
                 and mog2_mask is the cleaned binary MOG2 output.
        """
        if self.use_stabilization:
            frame, self.last_shift = self.stabilizer.stabilize(frame)
            
        # Apply passed exclusion mask to input frame BEFORE MOG2 and frame-diff
        if mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        if self.camera_mask is not None:
            # Mask out excluded regions BEFORE MOG2 processing
            frame = cv2.bitwise_and(frame, frame, mask=self.camera_mask)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame diff magnitude
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
        mag_map = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        raw_mask = self.mog2.apply(blurred, learningRate=self.learning_rate)

        # MOG2 with detectShadows=True marks shadow pixels as 127 (gray).
        # We only want true foreground (255), so threshold shadows out.
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Ensure mask is also cleaned post-MOG2 in case of artifacts
        if mask is not None:
            binary_mask = cv2.bitwise_and(binary_mask, mask)
            mag_map = cv2.bitwise_and(mag_map, mag_map, mask=mask)

        if self.camera_mask is not None:
            binary_mask = cv2.bitwise_and(binary_mask, self.camera_mask)
            mag_map = cv2.bitwise_and(mag_map, mag_map, mask=self.camera_mask)

        for (x1, y1, x2, y2) in self.ignore_regions:
            binary_mask[y1:y2, x1:x2] = 0
            mag_map[y1:y2, x1:x2] = 0

        return mag_map, binary_mask

    def reset(self):
        """Call when starting a new clip so stabilizer state doesn't leak across clips."""
        if self.stabilizer is not None:
            self.stabilizer.reset()


DEFAULT_DIFF_THRESHOLD = 20

def fuse_motion_signal(mag_map, mog2_mask, diff_threshold=DEFAULT_DIFF_THRESHOLD):
    """
    Fuses the frame-difference magnitude map and the MOG2 binary mask into a
    single binary motion signal using a logical OR (union).
    
    This ensures we do not gate out true incidents: a pixel is considered motion 
    if MOG2 flags it OR if the frame-difference exceeds the threshold.
    """
    import cv2
    _, diff_mask = cv2.threshold(mag_map, diff_threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_or(mog2_mask, diff_mask)


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
        mag_map, mask = estimator.get_motion_mask(frame)
        masks.append(mask)
        intensities.append(motion_intensity(mask))

    return masks, intensities

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
        mag_map, mask = estimator.get_motion_mask(frame)
        masks.append(mask)
        intensities.append(motion_intensity(mask))

    cap.release()

    metadata = {"fps": fps, "width": width, "height": height, "frame_count": frame_count}
    return masks, intensities, metadata