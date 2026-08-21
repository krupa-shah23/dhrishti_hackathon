"""
get_rois.py — frozen contract: get_rois(mask, grid_config) -> [boxes + seat_id]   (P4 owns this)

P4 writes the calibration config (see calibration_tool.py). P1 consumes this function,
calling it with her own motion mask + the config you hand her.

Matching logic:
  1. Find motion contours in the mask (same approach P1 already uses upstream —
     min-area filtering should match whatever threshold she's using; the 50px
     placeholder below is just a floor, tune it against her real numbers).
  2. For each contour, check if its center falls inside a known seat bbox
     (direct containment).
  3. If no seat bbox contains it (e.g. motion spans a gap between seats,
     or grid coverage is incomplete), fall back to nearest seat-center by
     distance rather than leaving seat_id as None — document this choice,
     since it's a real design decision, not an edge case to hide.
"""
import cv2
import json


def load_grid_config(path):
    with open(path) as f:
        return json.load(f)


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _contains(bbox, point):
    x1, y1, x2, y2 = bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def get_rois(mask, grid_config, min_area=50):
    """
    mask: binary motion mask (uint8, 0/255) — P1's MOG2/diff output
    grid_config: dict, either loaded via load_grid_config() or passed directly
                 {"camera_id": str, "grid_cells": [{"seat_id": str, "bbox": [x1,y1,x2,y2]}]}
    Returns: [{"seat_id": str | None, "bbox": [x1, y1, x2, y2]}]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_cells = grid_config.get("grid_cells", [])
    results = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        cx, cy = x + w / 2, y + h / 2

        matched_seat = None
        for cell in grid_cells:
            if _contains(cell["bbox"], (cx, cy)):
                matched_seat = cell["seat_id"]
                break

        if matched_seat is None and grid_cells:
            best_dist = float("inf")
            for cell in grid_cells:
                sx, sy = _center(cell["bbox"])
                dist = (sx - cx) ** 2 + (sy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    matched_seat = cell["seat_id"]

        results.append({
            "seat_id": matched_seat,
            "bbox": [x, y, x + w, y + h]
        })

    return results


if __name__ == "__main__":
    # Quick smoke test with a synthetic mask + config, no real video needed.
    import numpy as np

    test_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_mask, (20, 20), (60, 60), 255, -1)  # blob inside seat_01
    cv2.rectangle(test_mask, (150, 150), (190, 190), 255, -1)  # blob inside seat_02

    test_config = {
        "camera_id": "cam_test",
        "grid_cells": [
            {"seat_id": "seat_01", "bbox": [0, 0, 100, 100]},
            {"seat_id": "seat_02", "bbox": [100, 100, 200, 200]},
        ]
    }

    out = get_rois(test_mask, test_config)
    print("Smoke test result:", out)
    # findContours doesn't guarantee return order, so check by seat_id set, not position
    seat_ids_found = {r["seat_id"] for r in out}
    assert seat_ids_found == {"seat_01", "seat_02"}, f"Expected both seats tagged, got {seat_ids_found}"
    print("Smoke test passed.")