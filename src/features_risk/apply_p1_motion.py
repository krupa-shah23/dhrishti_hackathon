import pandas as pd


def aggregate_motion_for_event(motion_df, start_time, end_time, seat_id=None):
    """
    Aggregates P1's per-frame fused_motion_confidence/intensity into
    event-level avg_motion_intensity, peak_intensity, mog2_foreground_ratio.
    """
    window = motion_df[
        (motion_df["timestamp"] >= start_time) &
        (motion_df["timestamp"] <= end_time)
    ]
    if seat_id:
        window = window[window["seat_id"] == seat_id]

    if window.empty:
        return None

    return {
        "avg_motion_intensity": window["fused_motion_confidence"].mean(),
        "peak_intensity": window["fused_motion_confidence"].max(),
        "mog2_foreground_ratio": window["intensity"].mean(),
        "invigilator_excluded": bool(window["is_invigilator_motion"].any()),
        "seat_vacated_flag": bool(window["is_seat_vacated"].any()),
    }


def merge_p1_motion_into_gt(gt_csv_path, motion_csv_path, video_id, output_path=None):
    gt_df = pd.read_csv(gt_csv_path)
    motion_df = pd.read_csv(motion_csv_path)

    for col in ["avg_motion_intensity", "peak_intensity", "mog2_foreground_ratio"]:
        if col not in gt_df.columns:
            gt_df[col] = 0.0
    if "invigilator_excluded" not in gt_df.columns:
        gt_df["invigilator_excluded"] = False

    updated = 0
    for idx, row in gt_df[gt_df["video_id"] == video_id].iterrows():
        agg = aggregate_motion_for_event(motion_df, row["start_time"], row["end_time"])
        if agg:
            gt_df.at[idx, "avg_motion_intensity"] = agg["avg_motion_intensity"]
            gt_df.at[idx, "peak_intensity"] = agg["peak_intensity"]
            gt_df.at[idx, "mog2_foreground_ratio"] = agg["mog2_foreground_ratio"]
            gt_df.at[idx, "invigilator_excluded"] = agg["invigilator_excluded"]
            updated += 1

    out_path = output_path or gt_csv_path
    gt_df.to_csv(out_path, index=False)
    print(f"Updated {updated} ground-truth rows with real P1 motion data from {video_id}")
    return gt_df


if __name__ == "__main__":
    merge_p1_motion_into_gt(
        "../../data/ground_truth_labels.csv",
        "../../data/p3_input_data.csv",
        video_id="clip4",
    )