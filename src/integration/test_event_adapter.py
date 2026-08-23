import pytest
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_adapter import (
    adapt_bridge_event_to_schema,
    adapt_bridge_events_to_schema,
    adapt_bridge_event_to_node_payload,
    DEFAULT_EVENT_TYPE_PLACEHOLDER,
)
from src.integration.fastapi_bridge.schemas import Event


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
    schema_event = adapt_bridge_event_to_schema(bridge_event, video_id="test_vid_1")

    # Assert validation and field mappings
    assert isinstance(schema_event, Event)
    assert schema_event.event_id == "event_10"
    assert schema_event.video_id == "test_vid_1"
    assert schema_event.start == 0.0
    assert schema_event.end == 0.5
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
    bridge.process_fused_tracks(fused_tracks, frame_index=0)
    bridge.flush()

    events = bridge.get_completed_events()
    assert len(events) == 1
    bridge_event = events[0]
    assert bridge_event["object_detected"] is True

    schema_event = adapt_bridge_event_to_schema(
        bridge_event,
        video_id="test_vid_2",
        event_type="phone_detected",
        seat_id="seat_4B",
        notes="Detected near hand",
    )

    assert isinstance(schema_event, Event)
    assert schema_event.event_id == "event_25"
    assert schema_event.video_id == "test_vid_2"
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

    schema_events = adapt_bridge_events_to_schema(bridge_events, video_id="exam_hall_c1")
    assert len(schema_events) == 2
    assert all(isinstance(e, Event) for e in schema_events)
    assert schema_events[0].object_detected is True
    assert schema_events[1].object_detected is False


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


def test_node_payload_reshape_exact_shape():
    """
    Real ML event dict (produced by P2P3Bridge, not hand-fabricated) ->
    adapt_bridge_event_to_node_payload() -> asserts the exact JSON shape
    Node's POST /internal/events (src/routes/internal.js) expects, per
    src/models/Event.js: mlEventId, videoId, seatId, objectDetected (class
    name string), timestamps ([{start,end}]), bboxOverlay ([{frame,x,y,w,h}]).
    """
    bridge = P2P3Bridge(missing_threshold=5, fps=25.0)
    bridge.process_fused_tracks([
        {"track_id": 7, "box": (10, 20, 30, 40), "seat_id": "seat_3",
         "class": "phone", "confidence": 0.9},
    ], frame_index=0)
    bridge.process_fused_tracks([
        {"track_id": 7, "box": (12, 22, 32, 42), "seat_id": "seat_3",
         "class": "phone", "confidence": 0.6},
    ], frame_index=1)
    bridge.flush()

    bridge_events = bridge.get_completed_events()
    assert len(bridge_events) == 1
    event = bridge_events[0]

    payload = adapt_bridge_event_to_node_payload(event, video_id="video_abc123")

    assert set(payload.keys()) == {
        "mlEventId", "videoId", "seatId", "objectDetected", "objectConfidence",
        "timestamps", "bboxOverlay", "activities", "confidenceScore",
        "personIds", "duration",
    }
    # optional passthrough params omitted here -> schema-default-equivalent
    # values, except duration which is genuinely computable from the event
    # itself (end_time - start_time) regardless of caller.
    assert payload["activities"] == []
    assert payload["confidenceScore"] == 0.0
    assert payload["personIds"] == []
    assert payload["duration"] == round(event["end_time"] - event["start_time"], 2)
    # video_id-prefixed: bare event['event_id'] ("event_1") collides across
    # different videos since P2P3Bridge track_ids reset per pipeline run.
    assert payload["mlEventId"] == f"video_abc123_{event['event_id']}"
    assert payload["videoId"] == "video_abc123"
    assert payload["seatId"] == "seat_3"
    # objectDetected is the class-name string (best_detection's "class"),
    # not the ML-side bool/confidence -- see event_adapter.py docstring.
    assert payload["objectDetected"] == "phone"
    # objectConfidence: same best_detection dict's "confidence" (0.9, the
    # higher-confidence of the two frames fed into process_fused_tracks below).
    assert payload["objectConfidence"] == 0.9
    assert payload["timestamps"] == [{
        "start": event["start_time"],
        "end": event["end_time"],
    }]
    assert len(payload["bboxOverlay"]) == 2
    for box in payload["bboxOverlay"]:
        assert set(box.keys()) == {"frame", "x", "y", "w", "h"}
        assert "personId" not in box  # omitted, not faked -- no ReID linkage exists


def test_node_payload_requires_video_id():
    bridge = P2P3Bridge()
    bridge.process_fused_tracks([{"track_id": 1, "box": (0, 0, 10, 10)}], 0)
    bridge.flush()
    bridge_event = bridge.get_completed_events()[0]

    with pytest.raises(ValueError, match="video_id is required"):
        adapt_bridge_event_to_node_payload(bridge_event, video_id="")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
