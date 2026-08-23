# src/track_det/find_high_box_frames.py
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/spatial_cluster_subject1_3750_5400.json"
d = json.load(open(path))
pf = sorted(d["per_frame"], key=lambda e: -e["n_boxes"])

print("Top 10 highest-box-count frames in the window:")
for e in pf[:10]:
    print(f"  frame={e['frame']:>6}  n_boxes={e['n_boxes']}  n_clusters={e['n_clusters']}  "
          f"centroids={e['centroids']}")