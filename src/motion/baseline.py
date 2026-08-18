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
from .exclusion_regions import get_exclusion_mask
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


def calibrate_baseline(
    video_path: str,
    camera_id: str,
    *,
    calib_window_sec: Optional[float] = None,
    sample_rate: Optional[float] = None
) -> dict[str, tuple[float, float]]:
    """
    Streams through the calibration window of *video_path* and computes a
    per-seat motion-intensity baseline.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    camera_id : str
        Camera identifier used to look up grid and exclusion mask.
    calib_window_sec : float, optional
        Override for the calibration window length in seconds.
        If None, computed from the video duration via the master-plan rule.
    sample_rate : float, optional
        Frame sampling interval (1 = every frame).
        If None, derived from the video duration via the master-plan FPS policy.

    Returns
    -------
    dict mapping seat_id -> (mean, std)
    Returns an empty dict if the video cannot be opened or the camera is
    unknown.
    """
    # ---- 1. Open video to read metadata --------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if source_fps <= 0 or frame_count <= 0:
        return {}

    duration_sec = frame_count / source_fps

    # ---- 2. Resolve calibration window and sample rate -----------------
    if calib_window_sec is None:
        calib_window_sec = compute_calib_window(duration_sec)

    if sample_rate is None:
        target_fps = get_target_fps(duration_sec)
        sample_rate = calculate_sample_rate(source_fps, target_fps)

    # ---- 3. Load grid config for this camera ---------------------------
    grid = get_grid_config(camera_id, width, height)
    if not grid or not grid.get("seats"):
        return {}
    seats = grid["seats"]

    # ---- 4. Build camera exclusion mask --------------------------------
    cam_mask = get_exclusion_mask(camera_id, width, height)

    # ---- 5. Set up MotionEstimator (stateful MOG2) ---------------------
    estimator = MotionEstimator(camera_mask=cam_mask)

    # ---- 6. Accumulate per-seat intensity scalars ----------------------
    # Each seat accumulates a list of floats — one scalar per sampled frame.
    # No frame data is retained in memory.
    seat_samples: dict[str, list[float]] = {sid: [] for sid in seats}

    for frame_idx, timestamp_sec, frame in get_frame_stream_with_indices(
            video_path, sample_rate):

        # Stop at calibration window boundary
        if timestamp_sec > calib_window_sec:
            break

        motion_mask = estimator.get_motion_mask(frame)

        for seat_id, (x1, y1, x2, y2) in seats.items():
            intensity = _seat_intensity(motion_mask, x1, y1, x2, y2)
            seat_samples[seat_id].append(intensity)

    # ---- 7. Compute robust statistics per seat -------------------------
    result: dict[str, tuple[float, float]] = {}
    for seat_id, samples in seat_samples.items():
        mean, std = _robust_stats(samples)
        result[seat_id] = (mean, std)

    return result
