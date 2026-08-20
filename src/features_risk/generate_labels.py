import pandas as pd
from mock_track_data import generate_mock_tracks

def generate_label_rows(path="../../data/labels.csv", n_events=60):
    tracks = generate_mock_tracks(n_events)
    rows = [{
        "event_id": t["event_id"],
        "clip_name": t["clip_name"],
        "start_time": t["start_time"],
        "end_time": t["end_time"],
        "label": ""  # fill manually: Normal / Suspicious
    } for t in tracks]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} event rows at {path}")

if __name__ == "__main__":
    generate_label_rows()