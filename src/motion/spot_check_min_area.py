"""
spot_check_min_area.py
Owner: P1 - Motion & ROI

Validates that OEP_MIN_AREA=1500 doesn't strip real motion for a given
subject, mirroring the manual check already done for subject1 (frame 14000,
hand-to-glasses gesture). Auto-finds a real high-motion frame instead of
requiring a pre-known frame number, so it can be reused for subject10 (or
any future subject) without manual frame-hunting first.

Usage:
    python spot_check_min_area.py --video "data/OEP database/subject10/huangpi21.avi"
"""

import argparse
import cv2
import numpy as np

from motion import MotionEstimator
from roi import get_rois as _get_rois
def get_rois(*args, **kwargs):
    res = _get_rois(*args, **kwargs)
    if kwargs.get('return_cleaned'):
        return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
    return [b['bbox'] if isinstance(b, dict) else b for b in res], clean_mask

THRESHOLDS = (500, 1000, 1500, 2000)


def find_high_motion_frame(video_path, sample_stride=50):
    """
    Scans the clip (every `sample_stride` frames, for speed on long OEP
    videos) and returns the frame index + frame with the single largest
    contiguous foreground blob seen — a proxy for "a real, sizeable motion
    event", parallel to the manually-picked subject1 frame 14000.
    """
    estimator = MotionEstimator()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    best_idx, best_area, best_frame = -1, 0, None
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mag_map, mask = estimator.get_motion_mask(frame)
        if i % sample_stride == 0:
            cleaned = clean_mask(mask)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
                if max_area > best_area:
                    best_area, best_idx, best_frame = max_area, i, frame.copy()
        i += 1
    cap.release()
    return best_idx, best_frame, best_area


def check_thresholds(video_path, target_frame_idx):
    """
    Re-runs the clip up to target_frame_idx, then reports box count/sizes
    at each candidate min_area threshold on that exact frame — confirms
    whether the real motion box survives or gets filtered out.
    """
    estimator = MotionEstimator()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame = None
    mask = None
    for i in range(target_frame_idx + 1):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Video ended before reaching frame {target_frame_idx}")
        mag_map, mask = estimator.get_motion_mask(frame)
    cap.release()

    print(f"\n  Frame {target_frame_idx} — box count and sizes per min_area threshold:")
    for min_area in THRESHOLDS:
        boxes = get_rois(mask, min_area=min_area)
        sizes = [f"{(x2-x1)}x{(y2-y1)}" for (x1, y1, x2, y2) in boxes]
        print(f"    min_area={min_area:<5} -> {len(boxes)} box(es): {sizes}")

    return frame, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to the subject's webcam .avi file")
    ap.add_argument("--frame", type=int, default=None,
                     help="specific frame index to check (skips auto-search if given)")
    args = ap.parse_args()

    if args.frame is not None:
        target_idx = args.frame
        print(f"[+] Using manually specified frame {target_idx}")
    else:
        print(f"[+] Scanning for a high-motion frame in {args.video} ...")
        target_idx, _, area = find_high_motion_frame(args.video)
        if target_idx == -1:
            print("  [FAIL] No motion found in this clip at all — check the video file.")
            return
        print(f"  Found candidate frame {target_idx} (largest blob area: {area:.0f}px)")

    check_thresholds(args.video, target_idx)

    print(f"\n  Next: open outputs/masks/ overlay for this frame at min_area=1500 "
          f"(or rerun test_roi.py on this subject) and visually confirm the box "
          f"is on real motion, not noise, before trusting OEP_MIN_AREA=1500 for this subject.")


if __name__ == "__main__":
    main()