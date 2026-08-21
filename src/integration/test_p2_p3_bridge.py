import pytest
from src.integration.p2_p3_bridge import P2P3Bridge
from src.features_risk.extract_features import extract_features
from src.features_risk.train_risk_model import risk_score

def test_bridge_single_track():
    bridge = P2P3Bridge(missing_threshold=5, fps=10.0)

    # Frame 0
    fused_tracks_0 = [{
        "track_id": 1,
        "box": (10, 10, 30, 30),
        "class": "person",
        "confidence": 0.9,
        "invigilator_flag": False
    }]
    bridge.process_fused_tracks(fused_tracks_0, 0)
    assert len(bridge.active_tracks) == 1
    assert bridge.active_tracks[1]["start_frame"] == 0
    assert bridge.active_tracks[1]["end_frame"] == 0

    # Frame 1
    fused_tracks_1 = [{
        "track_id": 1,
        "box": (15, 15, 35, 35)
    }]
    bridge.process_fused_tracks(fused_tracks_1, 1)
    assert bridge.active_tracks[1]["end_frame"] == 1

    # End track
    bridge.flush()
    events = bridge.get_completed_events()
    assert len(events) == 1
    event = events[0]

    assert event["track_id"] == 1
    assert event["start_frame"] == 0
    assert event["end_frame"] == 1
    assert event["start_time"] == 0.0
    assert event["end_time"] == 0.1
    assert event["total_frames"] == 2

    assert event["object_detected"] is True
    assert event["is_invigilator"] is False

    # Correct XYXY -> XYWH
    assert event["boxes"] == [(10.0, 10.0, 20.0, 20.0), (15.0, 15.0, 20.0, 20.0)]

    # Correct centroid
    assert event["positions"] == [(20.0, 20.0), (25.0, 25.0)]

def test_bridge_empty_tracks_and_finalisation():
    bridge = P2P3Bridge(missing_threshold=2, fps=30.0)

    bridge.process_fused_tracks([{"track_id": 2, "box": (0, 0, 10, 10)}], 0)
    assert len(bridge.active_tracks) == 1
    assert len(bridge.get_completed_events()) == 0

    bridge.process_fused_tracks([], 1)
    assert len(bridge.active_tracks) == 1
    assert len(bridge.get_completed_events()) == 0

    bridge.process_fused_tracks([], 2)
    assert len(bridge.active_tracks) == 1

    bridge.process_fused_tracks([], 3)
    # At frame 3, last seen is 0. 3 - 0 = 3 > missing_threshold (2). Track should finalize.
    assert len(bridge.active_tracks) == 0
    events = bridge.get_completed_events()
    assert len(events) == 1

    event = events[0]
    assert event["track_id"] == 2
    assert event["start_frame"] == 0
    assert event["end_frame"] == 0

def test_bridge_missing_detector():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10)}], 0)
    bridge.flush()
    event = bridge.get_completed_events()[0]
    assert event["object_detected"] is None

def test_bridge_real_detector():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10), "class": "phone", "confidence": 0.8}], 0)
    bridge.flush()
    event = bridge.get_completed_events()[0]
    assert event["object_detected"] is True
    assert event["metadata"][0]["class"] == "phone"
    assert event["metadata"][0]["confidence"] == 0.8

def test_bridge_multiple_object_observations():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10)}], 0)
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10), "class": "phone"}], 1)
    bridge.flush()
    event = bridge.get_completed_events()[0]
    assert event["object_detected"] is True
    assert len(event["metadata"]) == 1
    assert event["metadata"][0]["frame"] == 1

def test_bridge_late_invigilator():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10), "invigilator_flag": False}], 0)
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10), "invigilator_flag": True}], 1)
    bridge.flush()
    event = bridge.get_completed_events()[0]
    assert event["is_invigilator"] is True

def test_p3_compatibility():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10)}], 0)
    bridge.process_fused_tracks([{"track_id": 1, "box": (5, 5, 15, 15)}], 1)
    bridge.flush()
    event = bridge.get_completed_events()[0]

    # Extract features using P3
    features = extract_features(event)

    expected_keys = [
        "motion_area", "duration", "frequency", "speed", "roi_size",
        "direction", "density", "persistence", "audio_energy",
        "onset_strength", "object_flag", "invigilator_flag",
        "phase_start", "phase_end"
    ]

    # Feature key names and ordering
    assert list(features.keys()) == expected_keys

    # Check return type of risk_score
    score = risk_score(features)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
