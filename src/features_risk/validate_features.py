import numpy as np
from mock_track_data import generate_mock_tracks
from extract_features import extract_features

def validate(n=60):
    tracks = generate_mock_tracks(n)
    bad = []
    for t in tracks:
        feats = extract_features(t)
        for k, v in feats.items():
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                bad.append((t["event_id"], k, v))
    if bad:
        print("FAILED — issues found:", bad)
    else:
        print(f"PASSED — all {n} events have clean feature dicts.")

if __name__ == "__main__":
    validate()