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
      - boxes: event['boxes'] (per-track list of per-frame (x, y, w, h) tuples)
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


def adapt_bridge_event_to_node_payload(
    event: Dict[str, Any],
    video_id: str,
    seat_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts a P2P3Bridge completed event dict into the exact JSON shape
    Node's POST /internal/events expects (src/models/Event.js), reshaping
    around three confirmed real-code mismatches:

      - objectDetected (Node: String class name, e.g. "phone") <- event['object_class'].
        NOT event['object_detected'] (bool) or event['object_confidence'] (float) --
        those have no Node destination and are intentionally dropped here.
        Decision: use the class-name string, not a new objectConfidence field,
        because detect_objects()'s per-detection "class" was already flowing
        through P2P3Bridge's metadata/best_detection and just needed
        surfacing (see p2_p3_bridge.py's "object_class" event key) -- no
        Node schema change was needed to get a real (non-guessed) value.
        objectConfidence (Node: Number, Event.js:49) <- event['object_confidence'],
        the same best_detection dict's "confidence" -- already surfaced at
        p2_p3_bridge.py:271, no new ML-side plumbing needed either.

      - timestamps (Node: [{start,end}] array) <- wraps event['start_time']/
        ['end_time'] as a single-element array. Node's schema (multiple
        ranges per event) is unchanged; ML only ever produces one range
        per finalized track-event, hence array-of-one.

      - bboxOverlay (Node: [{frame,x,y,w,h,personId}]) <- zipped from
        event['boxes'] and event['frame_indices'] (same length, same order --
        both appended together per-frame in P2P3Bridge.process_fused_tracks,
        see p2_p3_bridge.py:112). event['boxes'] tuples are ALREADY
        (x, y, w, h) -- built at p2_p3_bridge.py:77 as
        `p3_box = (x, y, w, h)` from the fused track's (x1,y1,x2,y2) box,
        specifically for P3/downstream consumers. (A prior verification
        pass incorrectly assumed these were still (x1,y1,x2,y2); that
        produced negative widths/heights here until caught by the real
        Node smoke test -- re-deriving w/h via subtraction was wrong.)
        personId is OMITTED per box, not faked: ML has no track_id->Person
        linkage (no ReID module exists in this repo -- confirmed absent;
        p2_p3_bridge.py:9-22 explicitly documents track_id as the only
        identity concept available, transient per-video). track_id itself
        cannot substitute here: Event.js's bboxSchema.personId
        (Event.js:23) is `mongoose.Schema.Types.ObjectId, ref: 'Person'` --
        a strict ref, not a loose string/int field -- so casting an int
        track_id into it would be a fabricated/invalid ObjectId, not a real
        reference. It has no `required: true`, so omitting it is valid
        against the real schema as-is. Populating it for real needs a
        Person document created per track (Re-ID or an interim
        track_id-keyed Person-record step), which is out of scope here.

    mlEventId (Node's idempotent upsert key, NOT Mongo _id) <-
        f"{video_id}_{event['event_id']}". video_id-prefixed, not bare
        event['event_id'] -- P2P3Bridge track_ids (and thus event_ids,
        "event_{track_id}") reset per pipeline run, so two different videos'
        first tracks both produce mlEventId "event_1". Caught live: bare
        event_id caused the real Node smoke test's second video's event to
        upsert onto (steal/reassign) the first video's event document. The
        video_id prefix is required for correctness, not just convention.
    """
    if not video_id:
        raise ValueError("video_id is required to build a Node event payload")

    boxes = event.get("boxes", [])  # already (x, y, w, h) -- see docstring
    frame_indices = event.get("frame_indices", [])
    bbox_overlay = []
    for (x, y, w, h), frame in zip(boxes, frame_indices):
        bbox_overlay.append({
            "frame": frame,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    return {
        "mlEventId": f"{video_id}_{event.get('event_id') or 'event_' + str(event.get('track_id', 0))}",
        "videoId": video_id,
        "seatId": seat_id if seat_id is not None else event.get("seat_ids", [None])[0] if event.get("seat_ids") else None,
        "objectDetected": event.get("object_class"),
        "objectConfidence": event.get("object_confidence"),
        "timestamps": [{
            "start": float(event.get("start_time", 0.0)),
            "end": float(event.get("end_time", 0.0)),
        }],
        "bboxOverlay": bbox_overlay,
    }


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
