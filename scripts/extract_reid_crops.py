"""
Extract a single-frame person crop from a video, given a frame index and
bounding box, for use in Re-ID threshold tuning (test_reid.py's
threshold_sweep function).

Usage (from repo root):
    uv run python scripts/extract_reid_crops.py \
        --video path/to/video1.mp4 --frame 452 --box 310 180 410 420 \
        --out data/debug_crops/reid_sweep/personA_video1.png

    uv run python scripts/extract_reid_crops.py \
        --video path/to/video2.mp4 --frame 118 --box 200 150 300 400 \
        --out data/debug_crops/reid_sweep/personA_video2.png

box is x1 y1 x2 y2 in pixel coordinates for that frame.

If you don't know the exact box yet, run with --frame only and
--preview to save the full frame first so you can eyeball coordinates
in an image viewer, then re-run with the box once you know it:
    uv run python scripts/extract_reid_crops.py \
        --video path/to/video1.mp4 --frame 452 --preview \
        --out data/debug_crops/reid_sweep/video1_frame452_full.png
"""
import argparse
from pathlib import Path

import cv2


def extract_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total_frames:
        cap.release()
        raise ValueError(f"frame {frame_index} out of range (video has {total_frames} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to extract")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                         help="Bounding box to crop, in pixel coords. Omit with --preview to save full frame instead.")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--preview", action="store_true",
                         help="Save the FULL frame (ignore --box) so you can eyeball coordinates first")
    args = parser.parse_args()

    frame = extract_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    print(f"Frame shape: {w}x{h} (width x height)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.preview or not args.box:
        cv2.imwrite(str(out_path), frame)
        print(f"Saved FULL frame preview to {out_path} — open it, find the person's "
              f"box coordinates (x1,y1,x2,y2), then re-run with --box.")
        return

    x1, y1, x2, y2 = args.box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box after clamping to frame bounds: ({x1},{y1},{x2},{y2})")

    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)
    print(f"Saved crop ({x2-x1}x{y2-y1}) to {out_path}")


if __name__ == "__main__":
    main()