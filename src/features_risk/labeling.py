import pandas as pd
import os


def create_labeling_sheet(path="../../data/labels.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(columns=["event_id", "clip_name", "start_time", "end_time", "label"])
    df.to_csv(path, index=False)
    print(f"Created labeling sheet at {path}")


def add_label_row(path, event_id, clip_name, start_time, end_time, label=""):
    df = pd.read_csv(path)
    new_row = {"event_id": event_id, "clip_name": clip_name,
               "start_time": start_time, "end_time": end_time, "label": label}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    create_labeling_sheet()