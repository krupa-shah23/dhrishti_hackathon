"""
roi.py
Owner: P1 - Motion & ROI

Shared contract (do not change signature without team sign-off):
    get_rois(mask) -> [boxes]

Box format LOCKED with P2: (x1, y1, x2, y2) — NOT (x, y, w, h).

Day-3 addition: apply_exclusion_mask() zeroes validated permanent-background
rectangles from the foreground mask BEFORE contour extraction.  Regions are
loaded from exclusion_regions.py and passed in by the caller — roi.py itself
remains clip-agnostic.
"""

import cv2 
import numpy as np

# Tune per clip type if needed; start with a conservative default.
MIN_CONTOUR_AREA = 500
MORPH_KERNEL_SIZE = (5, 5)
OPEN_KERNEL_SIZE = (7, 7)   # bigger than close kernel — erodes away grout-line/flicker noise before dilation can inflate it
DILATE_KERNEL_SIZE = (9, 9)  # bigger than morph kernel — merges fragmented body parts into one contour
DILATE_ITERATIONS = 2

MIN_ASPECT_RATIO = 0.25   # width/height — kills thin/wide slivers (grout lines, door edges)
MAX_ASPECT_RATIO = 1.8    # width/height — kills very flat wide blobs
MIN_FILL_RATIO = 0.35     # contour_area / bbox_area — kills sparse/scattered noise contours
MAX_AREA_FRACTION = 0.6   # bbox_area / frame_area — kills whole-frame blobs (exposure/lighting
                          # shifts, IR-cut switching) that min_area alone can never filter,
                          # since they're large, not small


def clean_mask(mask, kernel_size=MORPH_KERNEL_SIZE):
    """
    Removes salt-and-pepper noise via morphological open,
    then closes small gaps within real foreground blobs via close,
    then dilates to merge nearby fragments (e.g. head/torso/legs of one
    person being split into separate blobs) into a single contour.
    """
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL_SIZE)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_KERNEL_SIZE)
    dilated = cv2.dilate(closed, dilate_kernel, iterations=DILATE_ITERATIONS)

    return dilated


def is_valid_roi(area, w, h, min_area=MIN_CONTOUR_AREA, frame_area=None):
    """
    Filters out noise-shaped contours that survive area thresholding:
    thin slivers (grout lines, door edges), sparse/scattered blobs (flicker
    noise), and whole-frame blobs (exposure/lighting shifts) that aren't
    person-shaped.
    """
    if area < min_area:
        return False

    aspect_ratio = w / float(h) if h > 0 else 0
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return False

    bbox_area = w * h
    fill_ratio = area / float(bbox_area) if bbox_area > 0 else 0
    if fill_ratio < MIN_FILL_RATIO:
        return False

    if frame_area is not None and bbox_area > MAX_AREA_FRACTION * frame_area:
        return False  # whole-frame false positive (e.g. exposure/lighting shift), not real motion

    return True


def apply_exclusion_mask(mask: np.ndarray,
                         regions: list) -> np.ndarray:
    """
    Zeros out validated permanent-background rectangles from the foreground
    mask.  Called AFTER clean_mask() and BEFORE findContours so that noise
    originating from known static structures never reaches the contour stage.

    mask:    binary foreground mask (np.uint8, 0/255) — will NOT be mutated.
    regions: list of (x1, y1, x2, y2) tuples from exclusion_regions.py.
             Pass [] or omit the step for clips with no registered regions.
    returns: a new mask with the excluded rectangles zeroed out.
    """
    if not regions:
        return mask
    out = mask.copy()
    h_img, w_img = out.shape[:2]
    for (x1, y1, x2, y2) in regions:
        # Clip to frame bounds so an out-of-range entry can never raise.
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(x2, w_img - 1)
        cy2 = min(y2, h_img - 1)
        if cx2 > cx1 and cy2 > cy1:
            out[cy1:cy2 + 1, cx1:cx2 + 1] = 0
    return out


def get_rois(mask, min_area=MIN_CONTOUR_AREA, return_cleaned=False,
             exclusion_regions=None, camera_id=None):
    """
    mask: binary foreground mask (np.uint8, 0/255) from get_motion_mask()
    exclusion_regions: list of (x1, y1, x2, y2) validated background rects
        (from exclusion_regions.get_exclusion_regions(clip_name)).  If None
        or empty, no exclusion is applied — behaviour is identical to Day 2.
    returns: [ (x1, y1, x2, y2), ... ] — one box per valid motion region,
             already area- and shape-filtered. Empty list if no motion found.
    If return_cleaned=True, returns (boxes, cleaned_mask) instead, so callers
    that also want the cleaned mask (e.g. for debug saving) don't have to
    run clean_mask() a second time.
    """
    cleaned = clean_mask(mask)

    # --- Day-3 addition: zero out validated permanent-background zones -------
    if exclusion_regions:
        cleaned = apply_exclusion_mask(cleaned, exclusion_regions)
    # -------------------------------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = mask.shape[:2]
    frame_area = h_img * w_img
    
    seats_config = {}
    if camera_id:
        try:
            from .grid_config import get_grid_config
            grid = get_grid_config(camera_id, w_img, h_img)
            seats_config = grid.get("seats", {})
        except ImportError:
            pass

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if not is_valid_roi(area, w, h, min_area, frame_area=frame_area):
            continue  # drop noise-sized, noise-shaped, or whole-frame-blob contours before they leave this module
        x1, y1 = x, y
        x2 = min(x + w, w_img - 1)
        y2 = min(y + h, h_img - 1)
        bbox = (x1, y1, x2, y2)
        
        best_seat = "unknown"
        max_intersection = 0
        for seat_id, (sx1, sy1, sx2, sy2) in seats_config.items():
            ix1, iy1 = max(x1, sx1), max(y1, sy1)
            ix2, iy2 = min(x2, sx2), min(y2, sy2)
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection > max_intersection:
                    max_intersection = intersection
                    best_seat = seat_id
                    
        boxes.append({"seat_id": best_seat, "bbox": bbox})

    # Sort top-to-bottom, then left-to-right so box order is reproducible run-to-run
    boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    if return_cleaned:
        return boxes, cleaned
    return boxes


def draw_rois(frame, boxes, color=(0, 255, 0), thickness=2):
    """
    Debug helper: draws boxes on a copy of the frame for visual sanity checks.
    """
    out = frame.copy()
    for box_info in boxes:
        # Check if it's a dict (new format) or tuple (old format in some tests)
        if isinstance(box_info, dict):
            x1, y1, x2, y2 = box_info["bbox"]
            seat_id = box_info.get("seat_id", "unknown")
            cv2.putText(out, seat_id, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            x1, y1, x2, y2 = box_info
            
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out