# BLOCKED: bug in src/motion/grid_config.py (GRID_CONFIGS lacks configurations for Camera04, Mumbai04, LUCKNOW1, AH003)
"""
test_seat_id_passthrough.py

Runs grid_config.py seat lookup for candidate frames/images matching Camera04, Mumbai04, LUCKNOW1, AH003.
Asserts non-null seat_id returned.
Logs verification results to outputs/verification/seat_id_check.csv.
"""

import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.motion.grid_config import get_grid_config, GRID_CONFIGS

def lookup_seat_for_camera(camera_id: str, x: int, y: int, frame_w: int = 1280, frame_h: int = 720) -> str:
    """
    Performs seat lookup for a coordinate (x, y) on a camera frame via grid_config.py.
    Returns seat_id if matched, or None/'unknown' if unmapped.
    """
    config = get_grid_config(camera_id, target_width=frame_w, target_height=frame_h)
    seats = config.get("seats", {})
    if not seats:
        return None

    for seat_id, (sx1, sy1, sx2, sy2) in seats.items():
        if sx1 <= x < sx2 and sy1 <= y < sy2:
            return seat_id
    return None


class TestSeatIDPassthrough(unittest.TestCase):

    def test_seat_id_lookup_for_required_cameras(self):
        output_dir = os.path.join("outputs", "verification")
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, "seat_id_check.csv")

        target_cameras = ["Camera04", "Mumbai04", "LUCKNOW1", "AH003"]
        records = []
        missing_cameras = []

        # Find any images in frame_check/ or construct check entries
        frame_check_dir = "frame_check"
        image_files = []
        if os.path.exists(frame_check_dir):
            image_files = [f for f in os.listdir(frame_check_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for camera_id in target_cameras:
            # Check if camera_id exists in grid_config.py
            config_exists = camera_id in GRID_CONFIGS
            seat_id = lookup_seat_for_camera(camera_id, x=300, y=300, frame_w=1280, frame_h=720)

            records.append({
                "camera_id": camera_id,
                "config_present": config_exists,
                "seat_id_returned": seat_id if seat_id else "NONE_OR_UNKNOWN",
                "status": "PASS" if (config_exists and seat_id) else "BLOCKED_BUG_IN_GRID_CONFIG"
            })

            if not config_exists or not seat_id:
                missing_cameras.append(camera_id)

        df_out = pd.DataFrame(records)
        df_out.to_csv(csv_file, index=False)
        print(f"[INFO] Saved seat ID verification to {csv_file}")

        if missing_cameras:
            msg = (
                f"# BLOCKED: bug in src/motion/grid_config.py - "
                f"Missing camera configurations in GRID_CONFIGS for: {missing_cameras}"
            )
            print(f"[WARN] {msg}")
            # Assert failure to flag the real bug as instructed
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
