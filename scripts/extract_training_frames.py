"""
extract_training_frames.py

Pulls frames from your real exam-hall clips for phone/paper detector labeling.

Per the execution plan (P2 doc, Day 1): domain-matched frames from your own
camera/lighting/angle beat generic Kaggle/Roboflow volume. Target 150-300
labeled frames total, weighted toward the clips that actually show the
object.

Usage examples
--------------

# Extract ~1 fps from a whole clip
python scripts/extract_training_frames.py \
    --video "data/raw_clips/02.Candidate was found using a mobile.mp4" \
    --out data/frames_raw/clip02 \
    --fps 1

# Extract densely from just the window where the phone is visible
# (scrub the clip first, note rough start/end in seconds)
python scripts/extract_training_frames.py \
    --video "data/raw_clips/03.CCTV Mobile Usage.mkv" \
    --out data/frames_raw/clip03 \
    --fps 2 \
    --start 45 --end 120

# Batch mode: a JSON manifest of {video, out, fps, start, end} objects
python scripts/extract_training_frames.py --manifest clips_to_extract.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def extract_frames(video_path: str, out_dir: str, fps: float,
                    start: float | None, end: float | None) -> int:
    video_path = str(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] could not open: {video_path}", file=sys.stderr)
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps else 0

    start_frame = int((start or 0) * src_fps)
    end_frame = int((end if end is not None else duration) * src_fps)
    end_frame = min(end_frame, total_frames)

    # sample every Nth source frame to hit the target extraction fps
    step = max(1, round(src_fps / fps))

    stem = Path(video_path).stem.replace(" ", "_")
    saved = 0
    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            ts_sec = frame_idx / src_fps
            fname = f"{stem}_f{frame_idx:07d}_t{ts_sec:07.1f}s.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"  [OK] {video_path} -> {saved} frames "
          f"(src_fps={src_fps:.1f}, window={start or 0:.0f}s-"
          f"{end if end is not None else duration:.0f}s) -> {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="path to a single video file")
    ap.add_argument("--out", help="output directory for extracted frames")
    ap.add_argument("--fps", type=float, default=1.0,
                     help="target extraction rate (frames/sec), default 1.0")
    ap.add_argument("--start", type=float, default=None,
                     help="start time in seconds (default: 0)")
    ap.add_argument("--end", type=float, default=None,
                     help="end time in seconds (default: full clip)")
    ap.add_argument("--manifest", help="JSON file with a list of extraction jobs "
                                        "(overrides --video/--out/--fps/--start/--end)")
    args = ap.parse_args()

    jobs = []
    if args.manifest:
        with open(args.manifest) as f:
            jobs = json.load(f)
    elif args.video and args.out:
        jobs = [{
            "video": args.video, "out": args.out, "fps": args.fps,
            "start": args.start, "end": args.end,
        }]
    else:
        ap.error("provide either --video/--out or --manifest")

    total = 0
    print(f"Running {len(jobs)} extraction job(s)...")
    for job in jobs:
        total += extract_frames(
            job["video"], job["out"],
            job.get("fps", 1.0), job.get("start"), job.get("end"),
        )
    print(f"\nDone. {total} frames extracted total.")
    print("Next: label these in Roboflow or LabelImg (YOLO format), "
          "then run scripts/split_dataset.py to build train/val.")


if __name__ == "__main__":
    main()