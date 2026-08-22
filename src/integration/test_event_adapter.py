import pytest
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_adapter import (
    adapt_bridge_event_to_schema,
    adapt_bridge_events_to_schema,
    DEFAULT_EVENT_TYPE_PLACEHOLDER,
)
from src.integration.p2_field_resolvers import (
    resolve_camera_id,
    resolve_exam_phase,
    compute_repetition_count,
)
from src.integration.fastapi_bridge.schemas import Event


def test_resolve_camera_id():
    assert resolve_camera_id("clip7") == "cam_clip7"
    assert resolve_camera_id("07_seat_exchange") == "cam_clip7"
    assert resolve_camera_id("custom_video_1") == "cam_custom_video_1"
    assert resolve_camera_id("") == "cam_unknown"


def test_resolve_exam_phase():
    # video duration 100s
    assert resolve_exam_phase(5.0, 100.0) == "distribution"     # 5% <= 10%
    assert resolve_exam_phase(10.0, 100.0) == "distribution"    # 10% <= 10%
    assert resolve_exam_phase(50.0, 100.0) == "core"            # 50%
    assert resolve_exam_phase(90.0, 100.0) == "submission"      # 90% >= 90%
    assert resolve_exam_phase(95.0, 100.0) == "submission"      # 95% >= 90%
    # unknown duration
    assert resolve_exam_phase(5.0, 0.0) == "main_exam"
    assert resolve_exam_phase(5.0, -1.0) == "main_exam"


def test_compute_repetition_count():
    assert compute_repetition_count([]) == 0
    assert compute_repetition_count([False, False]) == 0
    assert compute_repetition_count([True, True, True]) == 1
    assert compute_repetition_count([False, True, True, False, False, True, False, True, True, True]) == 3
    assert compute_repetition_count([None, True, False, True, None]) == 2


def test_adapter_with_no_detection_event():
    """
    Validates that a real P2P3Bridge event without detection (object_detected=None)
    adapts cleanly to schemas.Event with object_detected=False.
    """
    bridge = P2P3Bridge(missing_threshold=5, fps=20.0)
    fused_tracks = [{
        "track_id": 10,
        "box": (50, 50, 150, 150),
        "class": None,
        "confidence": None,
        "invigilator_flag": False,
    }]
    bridge.process_fused_tracks(fused_tracks, frame_index=0)
    bridge.process_fused_tracks(fused_tracks, frame_index=10)
    bridge.flush()

    events = bridge.get_completed_events()
    assert len(events) == 1
    bridge_event = events[0]

    # Verify P2P3Bridge internal shape
    assert bridge_event["object_detected"] is None
    assert bridge_event["start_time"] == 0.0
    assert bridge_event["end_time"] == 0.5

    # Adapt to schema
    schema_event = adapt_bridge_event_to_schema(bridge_event, video_id="test_vid_1", video_duration=100.0)

    # Assert validation and field mappings
    assert isinstance(schema_event, Event)
    assert schema_event.event_id == "event_10"
    assert schema_event.video_id == "test_vid_1"
    assert schema_event.camera_id == "cam_test_vid_1"
    assert schema_event.start == 0.0
    assert schema_event.end == 0.5
    assert schema_event.exam_phase == "distribution"
    assert schema_event.repetition_count == 0
    assert schema_event.object_detected is False
    assert schema_event.object_confidence is None
    assert schema_event.event_type == DEFAULT_EVENT_TYPE_PLACEHOLDER
    assert schema_event.seat_id is None
    assert schema_event.notes is None

    # Verify Pydantic dump
    dumped = schema_event.model_dump()
    assert dumped["object_detected"] is False
    assert dumped["start"] == 0.0
    assert dumped["end"] == 0.5
    assert dumped["camera_id"] == "cam_test_vid_1"
    assert dumped["exam_phase"] == "distribution"


def test_adapter_with_detected_object_event():
    """
    Validates that a real P2P3Bridge event with object detection (object_detected=True)
    adapts cleanly to schemas.Event with object_detected=True and confidence.
    """
    bridge = P2P3Bridge(missing_threshold=5, fps=10.0)
    fused_tracks = [{
        "track_id": 25,
        "box": (100, 100, 200, 200),
        "class": "phone",
        "confidence": 0.92,
        "invigilator_flag": False,
    }]
    bridge.process_fused_tracks(fused_tracks, frame_index=500)
    bridge.flush()

    events = bridge.get_completed_events()
    assert len(events) == 1
    bridge_event = events[0]
    assert bridge_event["object_detected"] is True

    schema_event = adapt_bridge_event_to_schema(
        bridge_event,
        video_id="clip7",
        event_type="phone_detected",
        seat_id="seat_4B",
        notes="Detected near hand",
        video_duration=100.0,
    )

    assert isinstance(schema_event, Event)
    assert schema_event.event_id == "event_25"
    assert schema_event.video_id == "clip7"
    assert schema_event.camera_id == "cam_clip7"  # Mapped via CAMERA_ID_MAP
    assert schema_event.exam_phase == "core"      # 50s / 100s = 50% -> core
    assert schema_event.object_detected is True
    assert schema_event.object_confidence == 0.92
    assert schema_event.event_type == "phone_detected"
    assert schema_event.seat_id == "seat_4B"
    assert schema_event.notes == "Detected near hand"


def test_adapter_batch_events():
    """
    Validates batch conversion of multiple bridge events.
    """
    bridge = P2P3Bridge(missing_threshold=5, fps=10.0)
    bridge.process_fused_tracks([
        {"track_id": 1, "box": (0, 0, 10, 10), "class": "phone", "confidence": 0.8},
        {"track_id": 2, "box": (20, 20, 30, 30)},
    ], frame_index=0)
    bridge.flush()

    bridge_events = bridge.get_completed_events()
    assert len(bridge_events) == 2

    schema_events = adapt_bridge_events_to_schema(
        bridge_events,
        video_id="exam_hall_c1",
        video_duration=50.0,
    )
    assert len(schema_events) == 2
    assert all(isinstance(e, Event) for e in schema_events)
    assert schema_events[0].object_detected is True
    assert schema_events[1].object_detected is False
    assert schema_events[0].camera_id == "cam_exam_hall_c1"
    assert schema_events[0].exam_phase == "distribution"


def test_adapter_requires_video_id():
    """
    Validates that empty video_id raises ValueError.
    """
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10)}], 0)
    bridge.flush()
    bridge_event = bridge.get_completed_events()[0]

    with pytest.raises(ValueError, match="video_id is required"):
        adapt_bridge_event_to_schema(bridge_event, video_id="")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
