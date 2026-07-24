import numpy as np
import cv2  # type: ignore[import-not-found]

def get_motion_mask(frame: np.ndarray) -> np.ndarray:
    """
    Computes a binary foreground motion mask from a single video frame using MOG2.
    """
    raise NotImplementedError
