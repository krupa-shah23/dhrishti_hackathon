import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_overlap(start1, end1, start2, end2):
    """Fraction of overlap relative to the shorter event's duration."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0.0, overlap_end - overlap_start)
    shorter_duration = min(end1 - start1, end2 - start2)
    if shorter_duration <= 0:
        return 0.0
    return overlap / shorter_duration


def match_events(predicted_df, ground_truth_df, overlap_threshold=0.5):
    """
    predicted_df: columns [event_id, video_id, start_time, end_time, risk_label]
    ground_truth_df: columns [event_id, video_id, start_time, end_time, label]
    Returns y_true, y_pred lists (1=Suspicious, 0=Normal), matched by >=50% time overlap.
    """
    y_true, y_pred = [], []
    matched_pred_ids = set()

    for _, gt_row in ground_truth_df.iterrows():
        gt_label = 1 if gt_row["label"] == "Suspicious" else 0
        best_match = None
        best_overlap = 0.0

        candidates = predicted_df[predicted_df["video_id"] == gt_row["video_id"]]
        for _, pred_row in candidates.iterrows():
            if pred_row["event_id"] in matched_pred_ids:
                continue
            overlap = compute_overlap(
                gt_row["start_time"], gt_row["end_time"],
                pred_row["start_time"], pred_row["end_time"]
            )
            if overlap >= overlap_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_match = pred_row

        y_true.append(gt_label)
        if best_match is not None:
            pred_label = 1 if best_match["risk_label"] == "Suspicious" else 0
            y_pred.append(pred_label)
            matched_pred_ids.add(best_match["event_id"])
        else:
            y_pred.append(0)  # no match found = predicted Normal/missed

    return y_true, y_pred


def evaluate(predicted_csv, ground_truth_csv, overlap_threshold=0.5):
    predicted_df = pd.read_csv(predicted_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    y_true, y_pred = match_events(predicted_df, ground_truth_df, overlap_threshold)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Matched GT events: {len(y_true)}")
    return precision, recall, f1


if __name__ == "__main__":
    # Test now on synthetic data — logic proven before real labels arrive
    import pandas as pd
    from mock_event_data import generate_mock_events
    from extract_features_v2 import extract_features
    from severity_score import severity_score

    events = generate_mock_events(20)
    rows = []
    for e in events:
        feats = extract_features(e)
        score = severity_score(feats)
        label = "Suspicious" if score > 0.5 else "Normal"
        rows.append({
            "event_id": e["event_id"], "video_id": e["video_id"],
            "start_time": e["start_time"], "end_time": e["end_time"],
            "risk_label": label,
        })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv("../../outputs/test_predictions.csv", index=False)

    # reuse ground_truth_labels.csv (synthetic) to test matching logic
    evaluate("../../outputs/test_predictions.csv", "../../data/ground_truth_labels.csv")