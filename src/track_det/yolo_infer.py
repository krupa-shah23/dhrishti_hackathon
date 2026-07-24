import numpy as np

def detect_objects(frame: np.ndarray) -> list[tuple[list[int], str, float]]:
    """
    Detects prohibited objects using fine-tuned YOLO.
    Returns: list of (box, class_name, confidence)
    """
    raise NotImplementedError
