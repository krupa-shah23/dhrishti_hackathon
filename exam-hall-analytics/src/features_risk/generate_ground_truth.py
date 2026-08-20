import pandas as pd
import random
from mock_event_data import generate_mock_events

def generate_labels(n=50, path="../../data/ground_truth_labels.csv", suspicious_ratio=0.2):
    events = generate_mock_events(n)  # same seed=42, so IDs/times match exactly
    random.seed(99)  # separate seed just for label assignment

    rows = []
    for e in events:
        label = "Suspicious" if random.random() < suspicious_ratio else "Normal"
        rows.append({
            "event_id": e["event_id"],
            "video_id": e["video_id"],
            "start_time": e["start_time"],
            "end_time": e["end_time"],
            "seat_id": e["seat_id"],
            "label": label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} labeled events -> {path}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_labels(n=50)