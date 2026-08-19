import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features_risk.train_risk_model import risk_score
from src.features_risk.extract_features import extract_features
from src.features_risk.audio_utils import get_audio_energy, get_onset_strength

def run_test():
    print("Testing XGBoost Inference...")
    dummy_track = {
        "positions": [(10, 10), (12, 12)],
        "boxes": [(10, 10, 50, 50), (12, 12, 50, 50)],
        "start_time": 0.0,
        "end_time": 1.0,
        "total_frames": 25,
        "object_detected": True,
        "is_invigilator": False
    }
    features = extract_features(dummy_track)
    try:
        score = risk_score(features)
        print(f"SUCCESS: XGBoost risk score computed: {score:.4f}")
    except Exception as e:
        print(f"FAILED: XGBoost inference error: {e}")

    print("\nTesting Librosa Audio Extraction...")
    # Just need to check if the imports work and it doesn't crash on missing file
    try:
        energy = get_audio_energy("nonexistent.wav", 0.0, 1.0)
        print(f"SUCCESS: Audio energy fallback worked (value: {energy})")
    except Exception as e:
        print(f"FAILED: Audio extraction error: {e}")

if __name__ == "__main__":
    run_test()
