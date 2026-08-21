import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mock_event_data import generate_mock_events
from extract_features_v2 import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "severity_model.json")
_model = None

def build_training_data(labels_path="../../data/ground_truth_labels.csv"):
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df["label"].isin(["Normal", "Suspicious"])]

    X, y = [], []
    for _, row in labels_df.iterrows():
        event = {
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "seat_id": row.get("seat_id", "Desk_unknown"),
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration": row["end_time"] - row["start_time"],
            "avg_motion_intensity": row.get("avg_motion_intensity", 0.0) or 0.0,
            "peak_intensity": row.get("peak_intensity", 0.0) or 0.0,
            "mog2_foreground_ratio": row.get("mog2_foreground_ratio", 0.0) or 0.0,
            "repetition_count": 1,
            "object_detected": row.get("object_detected") if pd.notna(row.get("object_detected")) else None,
            "object_confidence": row.get("object_confidence", 0.0) or 0.0,
            "exam_phase": "mid",
        }
        feats = extract_features(event)
        X.append(list(feats.values()))
        y.append(1 if row["label"] == "Suspicious" else 0)

    feature_names = list(extract_features(event).keys())
    print("Feature names:", feature_names)
    print("First 5 rows of X:")
    for row_x in X[:5]:
        print(row_x)
    return np.array(X), np.array(y), feature_names

def train():
    X, y, feature_names = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("y_test:", y_test)
    print("preds:", preds)
    print(f"F1: {f1_score(y_test, preds, zero_division=0):.3f}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    return model

def load_model():
    global _model
    if _model is None:
        _model = XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model

def severity_score(features: dict) -> float:
    model = load_model()
    X = np.array([list(features.values())])
    return float(model.predict_proba(X)[0][1])

if __name__ == "__main__":
    train()