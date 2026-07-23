from mock_track_data import generate_mock_tracks
from extract_features import extract_features
from train_risk_model import risk_score

tracks = generate_mock_tracks(5)
for t in tracks:
    feats = extract_features(t)
    score = risk_score(feats)
    print(f"{t['event_id']}: risk_score = {score:.3f}")