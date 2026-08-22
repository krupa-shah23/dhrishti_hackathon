"""
Event adapter module for reconciling P2P3Bridge output dictionaries with
fastapi_bridge schemas.Event pydantic model.

P2P3Bridge._finalize_track() produces an internal event dictionary designed
for downstream P3 feature extraction (start_time, end_time, positions, boxes,
object_detected: True|None, is_invigilator, etc.).

This adapter maps that internal dictionary to a validated schemas.Event instance
for ingestion by the FastAPI webhook endpoint without altering P2P3Bridge's
internal representation.
"""
from typing import Dict, Any, Optional, List

try:
    from .fastapi_bridge.schemas import Event
except ImportError:
    from src.integration.fastapi_bridge.schemas import Event


# PLACEHOLDER: Default event_type until P1/P3's event-type taxonomy is finalized.
# Do not guess silently — this placeholder makes taxonomy dependence explicit.
DEFAULT_EVENT_TYPE_PLACEHOLDER = "unclassified"


def adapt_bridge_event_to_schema(
    event: Dict[str, Any],
    video_id: str,
    event_type: str = DEFAULT_EVENT_TYPE_PLACEHOLDER,
    seat_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Event:
    """
    Converts a single P2P3Bridge completed event dictionary into a valid schemas.Event instance.

    Mapping rules:
      - event_id: event['event_id'] (or derived as 'event_{track_id}')
      - video_id: required parameter (not present in raw P2 track event)
      - start: event['start_time']
      - end: event['end_time']
      - seat_id: seat_id parameter (default None; pending P1 seat mapping integration)
      - event_type: event_type parameter (default 'unclassified' placeholder)
      - object_detected: maps None -> False, True -> True (strict boolean)
      - object_confidence: event['object_confidence'] (float or None)
      - notes: notes parameter (default None)
      - boxes: event['boxes'] (per-track list of per-frame (x1,y1,x2,y2) tuples)
    """
    if not video_id:
        raise ValueError("video_id is required to create a valid schemas.Event")

    event_id = str(event.get("event_id") or f"event_{event.get('track_id', 0)}")
    start = float(event.get("start_time", 0.0))
    end = float(event.get("end_time", 0.0))

    # P2P3Bridge produces True or None; schemas.Event requires a strict bool
    raw_obj_det = event.get("object_detected")
    object_detected = bool(raw_obj_det) if raw_obj_det is not None else False

    object_confidence = event.get("object_confidence")
    if object_confidence is not None:
        object_confidence = float(object_confidence)

    boxes = [tuple(b) for b in event.get("boxes", [])]

    return Event(
        event_id=event_id,
        video_id=video_id,
        start=start,
        end=end,
        seat_id=seat_id,
        event_type=event_type,
        object_detected=object_detected,
        object_confidence=object_confidence,
        notes=notes,
        boxes=boxes,
    )


def adapt_bridge_events_to_schema(
    events: List[Dict[str, Any]],
    video_id: str,
    event_type: str = DEFAULT_EVENT_TYPE_PLACEHOLDER,
    seat_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> List[Event]:
    """
    Batch converts a list of P2P3Bridge completed event dictionaries into schemas.Event instances.
    """
    return [
        adapt_bridge_event_to_schema(
            event=e,
            video_id=video_id,
            event_type=event_type,
            seat_id=seat_id,
            notes=notes,
        )
        for e in events
    ]
