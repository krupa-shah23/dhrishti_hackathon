from mock_track_data import generate_mock_tracks
from extract_features import extract_features
from train_risk_model import risk_score

tracks = generate_mock_tracks(5)
for t in tracks:
    feats = extract_features(t)
    score = risk_score(feats)
    print(f"{t['event_id']}: risk={score:.3f}, invig={feats['invigilator_flag']}, "
          f"phase_start={feats['phase_start']}, phase_end={feats['phase_end']}, "
          f"audio_energy={feats['audio_energy']:.4f}")