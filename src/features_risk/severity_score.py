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
    events = generate_mock_events(len(labels_df) if len(labels_df) > 0 else 20)
    event_map = {e["event_id"]: e for e in events}

    X, y = [], []
    for _, row in labels_df.iterrows():
        e = event_map.get(row["event_id"])
        if e is None:
            continue
        feats = extract_features(e)
        X.append(list(feats.values()))
        y.append(1 if row["label"] == "Suspicious" else 0)
    feature_names = list(extract_features(events[0]).keys())
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