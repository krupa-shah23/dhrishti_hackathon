import numpy as np

def get_rois(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Extracts bounding boxes (ROIs) around motion regions from a binary mask.
    """
    raise NotImplementedError
