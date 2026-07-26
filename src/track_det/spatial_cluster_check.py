# src/track_det/spatial_cluster_check.py
"""
Spatial-clustering check for a fragmented ROI window.

Determines whether a burst of high track-ID fragmentation in a time window
is caused by (a) multiple real people producing spatially distinct box
clusters, or (b) a single person's motion being fragmented into many
overlapping/scattered boxes.

Usage:
    uv run python -m src.track_det.spatial_cluster_check \
        data/real_footage/subject1/subject1_rois.csv \
        --frame-start 3750 --frame-end 5400 \
        --distance-threshold 150.0 \
        --out outputs/spatial_cluster_subject1_3750_5400.json
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def load_rois(csv_path, frame_start, frame_end):
    rows_by_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            if frame_start <= fi <= frame_end:
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = x2 - x1, y2 - y1
                rows_by_frame[fi].append({"centroid": (cx, cy), "w": w, "h": h})
    return rows_by_frame


def cluster_frame(centroids, distance_threshold):
    """Cluster a single frame's centroids. Returns cluster labels (1-indexed)."""
    n = len(centroids)
    if n == 0:
        return []
    if n == 1:
        return [1]
    dists = pdist(np.array(centroids))
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")
    return labels.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--frame-end", type=int, required=True)
    ap.add_argument("--distance-threshold", type=float, default=150.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows_by_frame = load_rois(args.csv_path, args.frame_start, args.frame_end)

    if not rows_by_frame:
        print(f"No ROI rows found in frame range [{args.frame_start}, {args.frame_end}]. "
              f"Check the CSV path and column names (expected: frame_index,x1,y1,x2,y2).")
        return

    per_frame_cluster_counts = []
    per_frame_details = []
    box_widths, box_heights = [], []

    for fi in sorted(rows_by_frame.keys()):
        entries = rows_by_frame[fi]
        centroids = [e["centroid"] for e in entries]
        labels = cluster_frame(centroids, args.distance_threshold)
        n_clusters = len(set(labels)) if labels else 0
        per_frame_cluster_counts.append(n_clusters)
        per_frame_details.append({
            "frame": fi,
            "n_boxes": len(entries),
            "n_clusters": n_clusters,
            "cluster_labels": labels,
            "centroids": centroids,
        })
        box_widths.extend(e["w"] for e in entries)
        box_heights.extend(e["h"] for e in entries)

    counts = np.array(per_frame_cluster_counts)
    summary = {
        "frame_range": [args.frame_start, args.frame_end],
        "distance_threshold_px": args.distance_threshold,
        "n_active_frames": len(counts),
        "cluster_count_distribution": {
            str(k): int((counts == k).sum()) for k in sorted(set(counts.tolist()))
        },
        "mean_clusters_per_frame": float(counts.mean()),
        "median_clusters_per_frame": float(np.median(counts)),
        "std_clusters_per_frame": float(counts.std()),
        "box_width_range": [min(box_widths), max(box_widths)] if box_widths else None,
        "box_height_range": [min(box_heights), max(box_heights)] if box_heights else None,
    }

    print(json.dumps(summary, indent=2))

    print("\nInterpretation guide:")
    print("- If cluster_count_distribution is tightly concentrated at 2 (e.g. almost all "
          "frames show exactly 2 clusters) -> strong evidence of 2 real people.")
    print("- If it's concentrated at 1 -> boxes are spatially coherent; fragmentation is "
          "likely temporal/ID-churn, not multiple people.")
    print("- If it varies widely (1, 2, 3, 4+ across frames with no stable mode) -> "
          "consistent with noisy/scattered single-person fragmentation (hair/glasses/fabric "
          "texture triggering multiple disjoint contours), not clean multi-person structure.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_frame": per_frame_details}, f, indent=2)
        print(f"\nFull per-frame detail written to {args.out}")


if __name__ == "__main__":
    main()