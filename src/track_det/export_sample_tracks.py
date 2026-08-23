"""
export_sample_tracks.py

Runs track() on a real ROI CSV and accumulates each track's full history
across frames into the shape P3's extract_features() expects (positions,
boxes, start_time, end_time, total_frames). track() itself only returns
per-frame {"track_id", "box"} -- accumulation is done here, at the caller
level, since the contract doesn't expose full history directly.

Usage:
    uv run python -m src.track_det.export_sample_tracks data/real_footage/01_001/p1_rois_01_001.csv outputs/sample_tracks_01_001.json
"""
import sys
import json
import csv as csv_module
from pathlib import Path
from collections import defaultdict

from .tracker import track, reset_tracker


def load_rois(csv_path):
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frames[fi].append(box)
    return frames


def centroid(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def run(csv_path, out_path):
    reset_tracker()
    frames = load_rois(csv_path)
    max_frame = max(frames.keys())

    accum = {}  # track_id -> accumulator dict

    for f in range(0, max_frame + 1):
        boxes = frames.get(f, [])
        current_tracks = track(boxes)  # list of {"track_id": int, "box": (x1,y1,x2,y2)}

        for t in current_tracks:
            tid = t["track_id"]
            box = t["box"]
            if tid not in accum:
                accum[tid] = {
                    "track_id": tid,
                    "positions": [],
                    "boxes": [],
                    "start_time": f,
                    "end_time": f,
                    "total_frames": 0,
                }
            entry = accum[tid]
            entry["positions"].append(centroid(box))
            entry["boxes"].append(list(box))
            entry["end_time"] = f
            entry["total_frames"] += 1

    result = list(accum.values())
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Exported {len(result)} tracks to {out_path}")
    if result:
        sample = result[0]
        preview = {**sample, "positions": sample["positions"][:3] + (["..."] if len(sample["positions"]) > 3 else []),
                   "boxes": sample["boxes"][:3] + (["..."] if len(sample["boxes"]) > 3 else [])}
        print(f"Sample track (first one, truncated): {json.dumps(preview, indent=2)}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])