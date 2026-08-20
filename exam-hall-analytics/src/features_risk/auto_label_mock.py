import pandas as pd
from mock_track_data import generate_mock_tracks
from extract_features import extract_features

def auto_label(path="../../data/labels.csv"):
    tracks = generate_mock_tracks(80)
    rows = []
    for t in tracks:
        f = extract_features(t)
        score = f["speed"] * 0.3 + f["frequency"] * 50 + f["object_flag"] * 200
        label = "Suspicious" if score > 250 else "Normal"
        rows.append({
            "event_id": t["event_id"], "clip_name": t["clip_name"],
            "start_time": t["start_time"], "end_time": t["end_time"],
            "label": label
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Labeled {len(rows)} events -> {path}")

if __name__ == "__main__":
    auto_label()