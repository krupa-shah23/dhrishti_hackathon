from load_real_tracks import load_real_tracks
from extract_features import extract_features

tracks = load_real_tracks("../../data/sample_tracks_subject1.json")
for t in tracks[:3]:
    print(t["event_id"], extract_features(t))