import cv2
import numpy as np
from typing import Generator, Tuple

def get_target_fps(duration_sec: float) -> float:
    """
    Returns the target FPS based on the master plan duration-aware policy:
    < 30 s      -> 8.0 FPS
    30-120 s    -> 4.0 FPS
    120-600 s   -> 2.0 FPS
    > 600 s     -> 1.0 FPS
    """
    if duration_sec < 30.0:
        return 8.0
    elif duration_sec <= 120.0:
        return 4.0
    elif duration_sec <= 600.0:
        return 2.0
    else:
        return 1.0

def calculate_sample_rate(source_fps: float, target_fps: float) -> float:
    """
    Calculates the sample rate (interval) based on source and target FPS.
    If target_fps >= source_fps, returns 1.0 (no upsampling).
    """
    if target_fps <= 0 or source_fps <= 0:
        return 1.0
    if target_fps >= source_fps:
        return 1.0
    return source_fps / target_fps

def get_frame_stream_with_indices(path: str, camera_id: str = None, sample_rate: float = 1.0) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Genuinely streams frames from the video sequentially.
    Yields: (frame_index, timestamp_sec, frame)
    Uses a fractional-index approach to deterministically handle non-integer sample rates.
    Applies the exclusion mask at native resolution, then downsizes to 480p preserving aspect ratio.
    """
    from .exclusion_regions import get_exclusion_mask
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video stream: {path}")
        
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0 # Safe fallback for unknown streams
        
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    exclusion_mask = None
    if camera_id is not None:
        exclusion_mask = get_exclusion_mask(camera_id, orig_w, orig_h)
        
    target_h = 480
    target_w = orig_w
    if orig_h > 0 and orig_h != target_h:
        target_w = int(orig_w * (target_h / orig_h))
        
    # Ensure sample_rate is valid
    if sample_rate < 1.0:
        sample_rate = 1.0

    frame_index = 0
    next_target_index = 0.0
    startup_stable = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if not startup_stable:
                # Corrupted frames (e.g. missing I-frame) have extremely low variance.
                # Skip them so MOG2 doesn't learn a gray background.
                if np.std(frame) < 30.0:
                    frame_index += 1
                    continue
                startup_stable = True
                
            if frame_index >= next_target_index:
                timestamp_sec = frame_index / source_fps
                
                # 1. Apply mask at native resolution
                if exclusion_mask is not None:
                    frame = cv2.bitwise_and(frame, frame, mask=exclusion_mask)
                    
                # 2. Resize to target height
                if orig_h > 0 and orig_h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    
                yield (frame_index, timestamp_sec, frame)
                next_target_index += sample_rate
                
            frame_index += 1
    finally:
        cap.release()

def get_frame_stream(path: str, camera_id: str = None, sample_rate: float = 1.0) -> Generator[np.ndarray, None, None]:
    """
    Simple interface returning only the frame, wrapping the indexed stream.
    """
    for _, _, frame in get_frame_stream_with_indices(path, camera_id, sample_rate):
        yield frame
