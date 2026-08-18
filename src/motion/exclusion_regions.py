"""
exclusion_regions.py
Owner: P1 – Motion & ROI

Registry of per-clip static exclusion mask regions.

Each entry is a list of (x1, y1, x2, y2) rectangles that have been
VISUALLY VALIDATED as permanent background structures that never overlap
any zone where people normally appear or remain seated.

Approval protocol
-----------------
A region MUST satisfy ALL three criteria before it is added here:
  1. Permanently stationary (fixture, machine casing, wall, pipe, etc.)
  2. Does not overlap any walking path, seating area, or operating zone.
  3. Visual evidence inspected on representative frames spanning the full
     clip (early / mid / late).

Rejected candidate → Region 3 of copyMachine (200–268, 330–397):
  Overlaps a person operating the copy machine (confirmed at frame 2000).
  NOT included.
"""

import numpy as np

# Keys match the clip name keys used in test_roi.py CLIP_PATHS.
# Value: list of (x1, y1, x2, y2) rectangles to zero-out in the
# foreground mask before contour extraction.

EXCLUSION_REGIONS: dict[str, list[tuple[int, int, int, int]]] = {
    "copyMachine": [
        # Region 1: copy-machine paper-tray / indicator panel (top portion).
        # Verified at frame 80 — stationary machine structure, no human
        # overlap across the full clip.
        (104, 293, 139, 352),

        # Region 2: copy-machine base panel (bottom-left of machine).
        # Verified at frame 2800 — extreme edge, stationary, no human
        # overlap across the full clip.
        (0, 382, 56, 419),
    ],
}

def get_exclusion_regions(clip_name: str) -> list[tuple[int, int, int, int]]:
    """
    Returns the list of validated exclusion rectangles for a given clip.
    Returns an empty list for clips with no registered exclusions,
    so callers can always iterate without checking for None.
    """
    return EXCLUSION_REGIONS.get(clip_name, [])

# --- Camera-specific Exclusion Masks ---

CAMERA_CONFIGS = {
    "Camera12": {
        "resolution": (640, 480),
        "regions": [
            # Timestamp overlay (top-left)
            (30, 0, 295, 125),
            # Camera ID overlay (bottom-right)
            (410, 355, 525, 450)
        ]
    }
}

def get_exclusion_mask(camera_id: str, width: int, height: int) -> np.ndarray:
    """
    Returns a binary mask (255 = valid, 0 = excluded) for the given camera.
    Unknown cameras return an all-valid mask.
    Handles resolution matching/scaling safely.
    """
    mask = np.full((height, width), 255, dtype=np.uint8)
    
    if camera_id not in CAMERA_CONFIGS:
        return mask
        
    config = CAMERA_CONFIGS[camera_id]
    cfg_w, cfg_h = config["resolution"]
    
    scale_x = width / cfg_w if cfg_w > 0 else 1.0
    scale_y = height / cfg_h if cfg_h > 0 else 1.0
        
    for (x1, y1, x2, y2) in config["regions"]:
        sx1 = int(x1 * scale_x)
        sy1 = int(y1 * scale_y)
        sx2 = int(x2 * scale_x)
        sy2 = int(y2 * scale_y)
        
        # clamp to boundaries
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(width, sx2), min(height, sy2)
        
        if sx1 < sx2 and sy1 < sy2:
            mask[sy1:sy2, sx1:sx2] = 0
            
    return mask
