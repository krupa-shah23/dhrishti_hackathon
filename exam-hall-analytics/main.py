import sys
sys.path.append("src/features_risk")

from mock_track_data import generate_mock_tracks
from extract_features import extract_features
from train_risk_model import risk_score

def run_pipeline_demo(n_events=5):
    tracks = generate_mock_tracks(n_events)
    results = []
    for t in tracks:
        feats = extract_features(t)
        score = risk_score(feats)
        results.append({
            "event_id": t["event_id"],
            "clip_name": t["clip_name"],
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            **feats,
            "risk_score": score,
        })
        print(f"{t['event_id']}: risk_score={score:.3f}, "
              f"audio_energy={feats['audio_energy']:.4f}, "
              f"onset_strength={feats['onset_strength']:.4f}")
    return results

if __name__ == "__main__":
    run_pipeline_demo()