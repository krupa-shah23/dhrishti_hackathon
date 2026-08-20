import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mock_track_data import generate_mock_tracks
from extract_features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "risk_model.json")
_model = None


def build_training_data(labels_path="../../data/labels.csv"):
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df["label"].isin(["Normal", "Suspicious"])]

    tracks = generate_mock_tracks(len(labels_df) if len(labels_df) > 0 else 80)
    track_map = {t["event_id"]: t for t in tracks}

    X, y = [], []
    for _, row in labels_df.iterrows():
        t = track_map.get(row["event_id"])
        if t is None:
            continue
        feats = extract_features(t)
        X.append(list(feats.values()))
        y.append(1 if row["label"] == "Suspicious" else 0)

    feature_names = list(extract_features(tracks[0]).keys())
    return np.array(X), np.array(y), feature_names


def train():
    X, y, feature_names = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")

    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model


def load_model():
    global _model
    if _model is None:
        _model = XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model


def risk_score(features: dict) -> float:
    model = load_model()
    X = np.array([list(features.values())])
    proba = model.predict_proba(X)[0][1]
    return float(proba)


if __name__ == "__main__":
    train()