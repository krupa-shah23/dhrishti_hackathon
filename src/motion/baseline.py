"""
baseline.py
Owner: P1 - Motion & Calibration

Implements per-seat adaptive baseline calibration for the DRISHTI pipeline.

Interface
---------
calibrate_baseline(video_path, camera_id, calib_window_sec, grid_config,
                   camera_mask, sample_rate) -> dict[seat_id -> (mean, std)]

Design notes
------------
- Streams frames through the existing frame_stream API; never loads the
  whole video into RAM.
- The Camera12 exclusion mask is applied BEFORE any motion statistics are
  extracted, using the same MotionEstimator hook already established.
- Motion is measured PER SEAT as a fractional intensity:
      intensity = nonzero_pixels_in_seat_bbox / seat_area_pixels
  This is seat-local — never a global frame average.
- The signal currently measured is the MOG2 foreground mask, representing 
  the Stage-A motion evidence prior to fusion.
- Because the first 60 seconds of Camera12 is NOT an empty room, robust
  statistics are used: the Median and Median Absolute Deviation (MAD).
  These statistics are deterministically robust to outliers (such as brief
  candidate movements), ensuring the calibration represents the idle state.
- Coordinate convention: [x1, y1, x2, y2) matching grid_config.py.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .frame_stream import get_frame_stream_with_indices, get_target_fps, calculate_sample_rate
from .grid_config import get_grid_config
from .motion import MotionEstimator

# Master plan calibration window:
#   min(max(0.10 * duration_sec, 60), 300) seconds
MASTER_PLAN_CALIB_MIN_SEC = 60.0
MASTER_PLAN_CALIB_MAX_SEC = 300.0
MASTER_PLAN_CALIB_FRACTION = 0.10

def compute_calib_window(duration_sec: float) -> float:
    """
    Master-plan rule:
        calib_window = min(max(0.10 * duration_sec, 60), 300)
    """
    return min(max(MASTER_PLAN_CALIB_FRACTION * duration_sec,
                   MASTER_PLAN_CALIB_MIN_SEC),
               MASTER_PLAN_CALIB_MAX_SEC)


def _seat_intensity(motion_mask: np.ndarray, x1: int, y1: int,
                    x2: int, y2: int) -> float:
    """
    Returns the fraction of foreground pixels inside the seat bbox.
    Returns 0.0 for zero-area seats (safe fallback).
    Coordinates are [x1, y1, x2, y2) — upper-exclusive like Python slices.
    """
    area = max(1, (x2 - x1) * (y2 - y1))
    crop = motion_mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop)) / area


def _robust_stats(values: list[float]) -> tuple[float, float]:
    """
    Compute mean and std using robust estimators (Median and MAD) to handle 
    occupied calibration footage. Brief movements will be treated as outliers
    by the median, providing a stable baseline.

    Returns (mean, std).  Returns (0.0, 0.0) for empty input.
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    median = float(np.median(arr))
    # Median Absolute Deviation (MAD)
    mad = float(np.median(np.abs(arr - median)))
    # Approximate std dev for normal distribution = 1.4826 * MAD
    std = 1.4826 * mad
    return median, std


def collect_warmup_motion_masks(
    video_path: str,
    camera_id: str,
    *,
    calib_window_sec: Optional[float] = None,
    sample_rate: Optional[float] = None
) -> tuple[list[np.ndarray], dict]:
    """
    Streams through the calibration window of *video_path* and computes
    a list of motion masks (MOG2 foreground masks) and the grid dictionary.
    """
    # ---- 1. Open video to read metadata --------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if source_fps <= 0 or frame_count <= 0:
        return [], {}

    duration_sec = frame_count / source_fps

    # ---- 2. Resolve calibration window and sample rate -----------------
    if calib_window_sec is None:
        calib_window_sec = compute_calib_window(duration_sec)

    if sample_rate is None:
        target_fps = get_target_fps(duration_sec)
        sample_rate = calculate_sample_rate(source_fps, target_fps)

    # ---- 3. Load grid config for this camera ---------------------------
    # get_frame_stream now yields frames resized to 480p height.
    # We must fetch the grid config at the ACTUAL frame resolution (480p) we process,
    # scaling seat bounding boxes accordingly if they are stored at native res.
    target_h = 480
    target_w = width
    if height > 0 and height != target_h:
        target_w = int(width * (target_h / height))
        
    grid = get_grid_config(camera_id, target_w, target_h)
    if not grid or not grid.get("seats"):
        return [], {}

    # ---- 4. Set up MotionEstimator (stateful MOG2) ---------------------
    # Note: We do NOT request or apply a camera exclusion mask here. 
    # Exclusion masking is already applied at native resolution inside get_frame_stream.
    from .motion import MotionEstimator, fuse_motion_signal
    estimator = MotionEstimator()

    # ---- 5. Accumulate motion masks ------------------------------------
    masks = []

    for frame_idx, timestamp_sec, frame in get_frame_stream_with_indices(
            video_path, camera_id=camera_id, sample_rate=sample_rate):

        if timestamp_sec > calib_window_sec:
            break

        mag_map, mog2_mask = estimator.get_motion_mask(frame)
        fused_mask = fuse_motion_signal(mag_map, mog2_mask)
        masks.append(fused_mask)

    return masks, grid


def calibrate_baseline(
    warmup_frames: list[np.ndarray],
    grid: dict
) -> dict[str, tuple[float, float]]:
    """
    Computes a per-seat motion-intensity baseline over a sequence of 
    pre-computed warmup frames.

    Parameters
    ----------
    warmup_frames : list[np.ndarray]
        List of pre-computed motion masks (e.g. MOG2 foreground). These masks
        are yielded by get_frame_stream at a 480p downsampled resolution.
    grid : dict
        Grid configuration dictionary containing a 'seats' mapping.
        Note: The coordinates in this grid must already be scaled to match the
        480p resolution of the warmup_frames (derived via get_grid_config with target dimensions).

    Returns
    -------
    dict mapping seat_id -> (mean, std)
    Returns an empty dict if the grid contains no seats.
    """
    if not grid or not grid.get("seats"):
        return {}
        
    seats = grid["seats"]
    seat_samples: dict[str, list[float]] = {sid: [] for sid in seats}

    for motion_mask in warmup_frames:
        for seat_id, (x1, y1, x2, y2) in seats.items():
            intensity = _seat_intensity(motion_mask, x1, y1, x2, y2)
            seat_samples[seat_id].append(intensity)

    result: dict[str, tuple[float, float]] = {}
    for seat_id, samples in seat_samples.items():
        mean, std = _robust_stats(samples)
        result[seat_id] = (mean, std)

    return result

import collections

class RollingBaselineTracker:
    def __init__(self, initial_samples: dict[str, list[float]], window_size: int = 300):
        """
        Maintains a rolling window of per-seat scalar motion intensities.
        window_size: 300 frames (e.g., 2.5 minutes at 2 FPS).
        Maintains genuine rolling median/MAD to avoid the skew of mean-family estimators (like EMA)
        when real motion occurs inside the window.
        """
        self.window_size = window_size
        self.history = {
            seat_id: collections.deque(samples[-window_size:], maxlen=window_size)
            for seat_id, samples in initial_samples.items()
        }
        self._current_stats = {}
        self._recompute_all()
        
    def _recompute_all(self):
        for seat_id, samples in self.history.items():
            if samples:
                self._current_stats[seat_id] = _robust_stats(list(samples))
            else:
                self._current_stats[seat_id] = (0.0, 0.0)

    def update(self, seat_intensities: dict[str, float]) -> dict[str, tuple[float, float]]:
        """
        Adds new intensities and updates the baseline.
        Returns the updated median/mad stats.
        """
        for seat_id, intensity in seat_intensities.items():
            if seat_id not in self.history:
                self.history[seat_id] = collections.deque(maxlen=self.window_size)
            
            self.history[seat_id].append(intensity)
            # Recompute on the fly
            self._current_stats[seat_id] = _robust_stats(list(self.history[seat_id]))
            
        return self._current_stats
        
    def get_stats(self) -> dict[str, tuple[float, float]]:
        return self._current_stats

def calibrate_baseline_with_tracker(
    warmup_frames: list[np.ndarray],
    grid: dict,
    window_size: int = 300
) -> tuple[dict[str, tuple[float, float]], RollingBaselineTracker]:
    """
    Computes initial baseline and returns a RollingBaselineTracker initialized with the warmup samples.
    """
    if not grid or not grid.get("seats"):
        return {}, RollingBaselineTracker({}, window_size)
        
    seats = grid["seats"]
    seat_samples: dict[str, list[float]] = {sid: [] for sid in seats}

    for motion_mask in warmup_frames:
        for seat_id, (x1, y1, x2, y2) in seats.items():
            intensity = _seat_intensity(motion_mask, x1, y1, x2, y2)
            seat_samples[seat_id].append(intensity)

    tracker = RollingBaselineTracker(seat_samples, window_size=window_size)
    return tracker.get_stats(), tracker

