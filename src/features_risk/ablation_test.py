import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from mock_track_data import generate_mock_tracks
from extract_features import extract_features


def build_data(use_audio=True, use_invigilator=True, use_phase=True,
                labels_path="../../data/labels.csv"):
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
        if not use_audio:
            feats.pop("audio_energy", None)
            feats.pop("onset_strength", None)
        if not use_invigilator:
            feats.pop("invigilator_flag", None)
        if not use_phase:
            feats.pop("phase_start", None)
            feats.pop("phase_end", None)
        X.append(list(feats.values()))
        y.append(1 if row["label"] == "Suspicious" else 0)

    return np.array(X), np.array(y)


def run_config(use_audio, use_invigilator, use_phase, label):
    X, y = build_data(use_audio, use_invigilator, use_phase)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{label}: F1 = {f1:.3f}")
    return f1


if __name__ == "__main__":
    print("=== Ablation Table ===")
    results = []
    results.append(("Baseline (no audio, no invigilator, no phase)",
                     run_config(False, False, False, "Baseline")))
    results.append(("+ audio-fusion",
                     run_config(True, False, False, "+ audio-fusion")))
    results.append(("+ invigilator-filter",
                     run_config(False, True, False, "+ invigilator-filter")))
    results.append(("+ exam-phase-logic",
                     run_config(False, False, True, "+ exam-phase-logic")))
    results.append(("All combined",
                     run_config(True, True, True, "All combined")))

    df = pd.DataFrame(results, columns=["config", "f1"])
    df.to_csv("../../outputs/ablation_full.csv", index=False)
    print("\nLogged to outputs/ablation_full.csv")
    print(df)