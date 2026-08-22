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

# Mask coordinates are measured separately per resolution.
# Keys are (camera_id, (width, height)).
CAMERA_CONFIGS = {
    ("Camera12", (640, 480)): [
        # Timestamp overlay (top-left)
        (30, 0, 295, 125),
        # Camera ID overlay (bottom-right)
        (410, 355, 525, 450)
    ],
    ("Camera04", (1280, 720)): [
        # Top overlay (Date/Time)
        (20, 30, 550, 100),
        # Bottom overlay (Camera Label)
        (880, 620, 1150, 700)
    ],
    ("Camera12", (1280, 720)): [
        # Top-left overlay (Date/Time)
        (43, 0, 750, 115),
        # Bottom-right overlay (Camera Label)
        (951, 607, 1227, 676),
        # Bottom-right small timestamp overlay (just below label)
        (1150, 696, 1280, 720)
    ],
    ("AH003", (1280, 720)): [
        # Top-left overlay (Timestamp)
        (10, 30, 320, 80),
        # Bottom-right overlay (Label + '9')
        (900, 620, 1050, 670)
    ],
    ("DAHISAR1", (1280, 720)): [
        # Exclude invigilator desk (seat_7) from ROI detection
        # to prevent stationary high-motion from generating false incident events.
        (840, 160, 1110, 530)
    ]
}

def get_exclusion_mask(camera_id: str, width: int, height: int) -> np.ndarray:
    """
    Returns a binary mask (255 = valid, 0 = excluded) for the given camera and resolution.
    Unknown cameras or resolutions return an all-valid mask.
    """
    mask = np.full((height, width), 255, dtype=np.uint8)
    
    key = (camera_id, (width, height))
    if key not in CAMERA_CONFIGS:
        return mask
        
    regions = CAMERA_CONFIGS[key]
        
    for (x1, y1, x2, y2) in regions:
        # clamp to boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 0
            
    return mask
