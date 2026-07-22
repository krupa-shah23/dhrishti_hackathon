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
    # Add future validated clip entries below, same format.
    # Example:
    # "cubicle": [
    #     (x1, y1, x2, y2),
    # ],
}


def get_exclusion_regions(clip_name: str) -> list[tuple[int, int, int, int]]:
    """
    Returns the list of validated exclusion rectangles for a given clip.
    Returns an empty list for clips with no registered exclusions,
    so callers can always iterate without checking for None.
    """
    return EXCLUSION_REGIONS.get(clip_name, [])
