"""
scan_fullframe_blobs.py
Owner: P1 - Motion & ROI

Scans a video for frames where a single ROI box covers most of the frame
(exposure/lighting-shift false positive, distinct from ordinary noise —
min_area filtering can never catch this since the blob is large, not small).
Run BEFORE and AFTER the MAX_AREA_FRACTION fix in roi.py to confirm the fix
actually eliminates the pattern rather than just reducing box count.

Usage:
    python scan_fullframe_blobs.py --video "data/OEP database/subject10/huangpi21.avi"
"""

import argparse
import cv2

from motion import MotionEstimator
from roi import get_rois as _get_rois
def get_rois(*args, **kwargs):
    res = _get_rois(*args, **kwargs)
    if kwargs.get('return_cleaned'):
        return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
    return [b['bbox'] if isinstance(b, dict) else b for b in res]

FRACTION_THRESHOLD = 0.6  # matches MAX_AREA_FRACTION in roi.py


def scan(video_path, min_area=1500):
    estimator = MotionEstimator()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = 0
    fullframe_frames = 0
    fullframe_indices = []

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame_area = h * w

        mag_map, mask = estimator.get_motion_mask(frame)
        boxes = get_rois(mask, min_area=min_area)

        hit = False
        for (x1, y1, x2, y2) in boxes:
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > FRACTION_THRESHOLD * frame_area:
                hit = True
                break

        if hit:
            fullframe_frames += 1
            fullframe_indices.append(i)

        total_frames += 1
        i += 1

    cap.release()

    pct = (fullframe_frames / total_frames * 100) if total_frames else 0
    print(f"\nTotal frames scanned : {total_frames}")
    print(f"Full-frame-blob hits : {fullframe_frames} ({pct:.2f}%)")
    if fullframe_indices:
        preview = fullframe_indices[:20]
        print(f"First occurrences (up to 20): {preview}")
        if len(fullframe_indices) > 20:
            print(f"  ... and {len(fullframe_indices) - 20} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--min_area", type=int, default=1500)
    args = ap.parse_args()
    scan(args.video, min_area=args.min_area)


if __name__ == "__main__":
    main()