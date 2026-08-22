#!/usr/bin/env python3
"""
dedup_frames.py — Near-duplicate frame thinning (DRISHTI, P2a Phase 1)

Collapses a folder of consecutively-extracted video frames down to the visually
distinct candidates before manual labeling review. Consecutive frames from one
continuous "phone hidden" stretch are near-identical — this keeps only frames
that differ meaningfully (perceptual hash distance) from the last KEPT frame,
so you're not hand-reviewing all 876.

Setup (if not already installed):
    pip install imagehash pillow --break-system-packages

Usage:
    python dedup_frames.py --input data/frames_raw/new_clip --output data/frames_raw/new_clip_deduped

Nothing here touches your source folder — it only copies kept frames into --output
and writes a manifest CSV so you can sanity-check the threshold before trusting it.
Cheap to re-run with a different --threshold if you keep too many or too few frames.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

import imagehash
from PIL import Image


def natural_sort_key(path: Path):
    """Sort frame_1.png, frame_2.png, ..., frame_10.png in numeric order, not lexical."""
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p for p in parts]


def dedup_frames(input_dir: Path, output_dir: Path, threshold: int, extensions: set):
    frames = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions],
        key=natural_sort_key,
    )
    if not frames:
        raise SystemExit(f"No frames found in {input_dir} with extensions {extensions}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dedup_manifest.csv"

    kept_hash = None
    kept_count = 0
    rows = []

    for frame_path in frames:
        img_hash = imagehash.phash(Image.open(frame_path))

        if kept_hash is None:
            distance = None
            keep = True
        else:
            distance = img_hash - kept_hash  # Hamming distance, 0-64
            keep = distance >= threshold

        if keep:
            shutil.copy2(frame_path, output_dir / frame_path.name)
            kept_hash = img_hash
            kept_count += 1

        rows.append({
            "frame": frame_path.name,
            "kept": keep,
            "hash_distance_from_last_kept": distance,
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "kept", "hash_distance_from_last_kept"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input frames:  {len(frames)}")
    print(f"Kept frames:   {kept_count}")
    print(f"Dropped:       {len(frames) - kept_count}")
    print(f"Manifest:      {manifest_path}")
    print(f"Kept frames copied to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Thin near-duplicate frames before manual labeling review.")
    parser.add_argument("--input", required=True, type=Path, help="Folder of raw extracted frames")
    parser.add_argument("--output", required=True, type=Path, help="Folder to write kept frames + manifest")
    parser.add_argument(
        "--threshold", type=int, default=8,
        help="Minimum perceptual-hash Hamming distance from the last kept frame to keep a new frame. "
             "Higher = more aggressive thinning. Start at 8 (default) — if you end up with far more or "
             "fewer than the ~150-250 target, re-run with a different value, it's cheap."
    )
    parser.add_argument(
        "--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"],
        help="Frame file extensions to include"
    )
    args = parser.parse_args()

    dedup_frames(args.input, args.output, args.threshold, set(e.lower() for e in args.extensions))


if __name__ == "__main__":
    main()