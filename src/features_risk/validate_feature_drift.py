import numpy as np
import pandas as pd
from extract_features import extract_features


# Reference ranges from CDNet-based mock training data (update after checking your real trained set)
CDNET_RANGES = {
    "motion_area": (0, 6000),
    "duration": (0, 8),
    "speed": (0, 400),
    "density": (0, 1),
}


def check_drift(shanghaitech_tracks: list):
    """
    Pass in real track objects once P2 delivers ShanghaiTech-derived tracks.
    Flags features whose values fall clearly outside CDNet-trained ranges.
    """
    rows = []
    for t in shanghaitech_tracks:
        feats = extract_features(t)
        rows.append(feats)

    df = pd.DataFrame(rows)

    print("=== ShanghaiTech Feature Distributions ===")
    print(df[["motion_area", "duration", "speed", "density"]].describe())

    print("\n=== Drift Check vs CDNet Ranges ===")
    for feat, (lo, hi) in CDNET_RANGES.items():
        out_of_range = df[(df[feat] < lo) | (df[feat] > hi)]
        pct = len(out_of_range) / len(df) * 100 if len(df) > 0 else 0
        status = "OK" if pct < 20 else "DRIFT - investigate"
        print(f"{feat}: {pct:.1f}% out of CDNet range ({lo}-{hi}) -> {status}")

    df.to_csv("../../outputs/shanghaitech_feature_check.csv", index=False)
    print("\nSaved to outputs/shanghaitech_feature_check.csv")
    return df


if __name__ == "__main__":
    # Placeholder — replace with real tracks once P2 delivers them
    print("Waiting on P2's ShanghaiTech-derived tracks. Run this with real track list once available:")
    print("  from validate_feature_drift import check_drift")
    print("  check_drift(real_shanghaitech_tracks)")