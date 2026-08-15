"""
heatmap.py — P4: Outputs, Evaluation & Integration
Accumulates per-frame motion masks into a spatial heatmap and overlays
it onto the original frame using cv2.COLORMAP_JET.
"""

import cv2  # type: ignore[import-not-found]
import numpy as np


def accumulate_mask(accumulator: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Adds a new binary motion mask to the running accumulator.

    Args:
        accumulator (np.ndarray): Float32 array (H, W) accumulated so far.
        mask (np.ndarray): Binary motion mask (H, W), values 0 or 255.

    Returns:
        np.ndarray: Updated accumulator (H, W, float32).
    """
    return accumulator + (mask.astype(np.float32) / 255.0)


def generate_heatmap(accumulator: np.ndarray) -> np.ndarray:
    """
    Converts the accumulated motion mask into a color heatmap image.

    Args:
        accumulator (np.ndarray): Float32 accumulated mask (H, W).

    Returns:
        np.ndarray: BGR heatmap image (H, W, 3).
    """
    if accumulator.max() == 0:
        # Return a blank heatmap if no motion detected at all
        h, w = accumulator.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Normalize to [0, 255]
    normalized = cv2.normalize(accumulator, np.empty_like(accumulator), 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # Apply JET colormap: cool (blue) = low motion, hot (red) = high motion
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends the heatmap onto the original frame using weighted addition.

    Args:
        frame (np.ndarray): Original BGR video frame (H, W, 3).
        heatmap (np.ndarray): BGR heatmap image (H, W, 3).
        alpha (float): Heatmap opacity [0.0 – 1.0]. Default 0.5.

    Returns:
        np.ndarray: Blended BGR image (H, W, 3).
    """
    # Resize heatmap to match frame if dimensions differ
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_heatmap(heatmap: np.ndarray, output_path: str) -> None:
    """
    Saves the heatmap image to disk.

    Args:
        heatmap (np.ndarray): BGR heatmap image.
        output_path (str): Full output file path (e.g., 'outputs/video_heatmap.png').
    """
    cv2.imwrite(output_path, heatmap)
    print(f"[heatmap] Saved -> {output_path}")
