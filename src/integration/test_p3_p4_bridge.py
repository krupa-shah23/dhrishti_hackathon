# src/integration/test_p3_p4_bridge.py
import os
from src.integration.p3_p4_bridge import p3_event_to_csv_row
from src.outputs_eval.logger import init_log, log_event

from src.integration.p3_p4_bridge import p3_events_to_timeline_arrays


def test_p3_p4_bridge_basic():
    # Fake completed P2P3Bridge event (shape matches active_tracks dict in p2_p3_bridge.py)
    fake_event = {
        "event_id": "event_1",
        "track_id": 1,
        "boxes": [(100.0, 80.0, 100.0, 320.0)],  # (x, y, w, h)
        "start_frame": 0,
        "end_frame": 45,
        "metadata": [{"class": "phone", "confidence": 0.87}],
    }

    # Fake P3 features (shape matches extract_features() output — only fields we use here)
    fake_features = {
        "motion_area": 32000.0,   # w*h from the box above
        "audio_energy": 12.4,
    }

    fake_risk = 0.73
    fps = 25.0
    frame_width, frame_height = 640, 480

    row = p3_event_to_csv_row(fake_event, fake_features, fake_risk, fps, frame_width, frame_height)

    print(row)

    # Sanity checks
    assert row["frame_index"] == 45
    assert row["roi_box"] == "(100,80,100,320)"
    assert 0.0 <= row["motion_score"] <= 100.0
    assert row["confidence"] == 0.87
    assert row["risk_score"] == 0.73

    # Write it through the real logger to a throwaway CSV and eyeball it
    csv_path = "outputs/test_p3_p4_bridge_events.csv"
    init_log(csv_path)
    log_event(csv_path, row)

    with open(csv_path) as f:
        print(f.read())

    assert os.path.exists(csv_path)

# append to src/integration/test_p3_p4_bridge.py, or run standalone


def test_timeline_arrays():
    fake_event1 = {"end_frame": 45}
    fake_features1 = {"motion_area": 32000.0}
    fake_event2 = {"end_frame": 100}
    fake_features2 = {"motion_area": 15000.0}

    events_with_scores = [
        (fake_event1, fake_features1, 0.73),
        (fake_event2, fake_features2, 0.20),
    ]

    ts, motion, risk = p3_events_to_timeline_arrays(events_with_scores, fps=25.0, frame_width=640, frame_height=480)
    print(ts, motion, risk)
    assert len(ts) == len(motion) == len(risk) == 2
    assert ts == sorted(ts)

if __name__ == "__main__":
    test_timeline_arrays()

if __name__ == "__main__":
    test_p3_p4_bridge_basic()