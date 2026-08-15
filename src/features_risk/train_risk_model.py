import os
import numpy as np
from xgboost import XGBClassifier

from .extract_features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "risk_model.json")
_model = None


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
