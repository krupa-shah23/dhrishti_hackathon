from extract_features_v2 import extract_features
from severity_score import severity_score

def test_edge_cases():
    edge_events = [
        {  # all-zero event
            "event_id": "EDGE001", "start_time": 0.0, "end_time": 0.0,
            "duration": 0.0, "avg_motion_intensity": 0.0, "peak_intensity": 0.0,
            "repetition_count": 0, "object_detected": None,
            "mog2_foreground_ratio": 0.0, "exam_phase": "mid",
        },
        {  # extreme values
            "event_id": "EDGE002", "start_time": 50.0, "end_time": 150.0,
            "duration": 100.0, "avg_motion_intensity": 9999.0, "peak_intensity": 9999.0,
            "repetition_count": 50, "object_detected": "phone",
            "mog2_foreground_ratio": 1.0, "exam_phase": "start",
        },
        {  # roi_size = 0 (division-by-zero guard test)
            "event_id": "EDGE003", "start_time": 10.0, "end_time": 12.0,
            "duration": 2.0, "avg_motion_intensity": 50.0, "peak_intensity": 0.0,
            "repetition_count": 1, "object_detected": None,
            "mog2_foreground_ratio": 0.5, "exam_phase": "end",
        },
        {  # missing fields entirely
            "event_id": "EDGE004", "start_time": 5.0, "end_time": 8.0,
        },
    ]

    print("=== Edge Case Test ===")
    for e in edge_events:
        try:
            feats = extract_features(e)
            score = severity_score(feats)
            print(f"{e['event_id']}: OK, score={score:.3f}")
        except Exception as ex:
            print(f"{e['event_id']}: FAILED — {ex}")

if __name__ == "__main__":
    test_edge_cases()