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
from __future__ import annotations
from typing import Dict, Any, Optional, List

try:
    from .fastapi_bridge.schemas import Event, BBoxOverlayEntry
    from .p2_field_resolvers import (
        resolve_camera_id,
        resolve_exam_phase,
        compute_repetition_count,
    )
    from .thumbnail_heatmap_writer import save_event_thumbnail, save_event_heatmap
    from .reasoning_layer import get_llm_reasoning
except ImportError:
    from src.integration.fastapi_bridge.schemas import Event, BBoxOverlayEntry
    from src.integration.p2_field_resolvers import (
        resolve_camera_id,
        resolve_exam_phase,
        compute_repetition_count,
    )
    from src.integration.thumbnail_heatmap_writer import save_event_thumbnail, save_event_heatmap
    from src.integration.reasoning_layer import get_llm_reasoning


# PLACEHOLDER: Default event_type until P1/P3's event-type taxonomy is finalized.
# Do not guess silently — this placeholder makes taxonomy dependence explicit.
DEFAULT_EVENT_TYPE_PLACEHOLDER = "unclassified"


def adapt_bridge_event_to_schema(
    event: Dict[str, Any],
    video_id: str,
    event_type: str = DEFAULT_EVENT_TYPE_PLACEHOLDER,
    seat_id: Optional[str] = None,
    notes: Optional[str] = None,
    video_duration: Optional[float] = None,
) -> Event:
    """
    Converts a single P2P3Bridge completed event dictionary into a valid schemas.Event instance.

    Mapping rules:
      - event_id: event['event_id'] (or derived as 'event_{track_id}')
      - video_id: required parameter (not present in raw P2 track event)
      - camera_id: resolved via resolve_camera_id(video_id) or event['camera_id']
      - start: event['start_time']
      - end: event['end_time']
      - duration: end - start
      - exam_phase: resolved via resolve_exam_phase(start, video_duration)
      - seat_id: seat_id parameter (default None; pending P1 seat mapping integration)
      - event_type: event_type parameter (default 'unclassified' placeholder)
      - object_detected: maps None -> False, True -> True (strict boolean)
      - object_confidence: event['object_confidence'] (float or None)
      - repetition_count: computed via compute_repetition_count() if per-frame data available, else 0
      - invigilator_excluded: event['is_invigilator'] (bool)
      - person_ids: [f"person_{track_id}"] or event['person_ids']
      - activities: [event_type]
      - bbox_overlay: constructed from event['boxes'] and event['frame_indices'] if present
      - notes: notes parameter (default None)
    """
    if not video_id:
        raise ValueError("video_id is required to create a valid schemas.Event")

    event_id = str(event.get("event_id") or f"event_{event.get('track_id', 0)}")
    start = float(event.get("start_time", 0.0))
    end = float(event.get("end_time", 0.0))
    duration = float(event.get("duration", max(0.0, end - start)))

    # P2P3Bridge produces True or None; schemas.Event requires a strict bool
    raw_obj_det = event.get("object_detected")
    object_detected = bool(raw_obj_det) if raw_obj_det is not None else False

    object_confidence = event.get("object_confidence")
    if object_confidence is not None:
        object_confidence = float(object_confidence)

    # Invigilator exclusion from P2P3Bridge flag
    invigilator_excluded = bool(event.get("is_invigilator", False))

    # Person IDs from track_id
    track_id = event.get("track_id")
    person_ids = event.get("person_ids") or ([f"person_{track_id}"] if track_id is not None else [])

    # Activities list
    activities = event.get("activities") or ([event_type] if event_type else [])

    # Severity & Color tag
    color_tag = "red" if object_detected else "yellow"

    # Construct bbox_overlay entries if boxes are present in P2P3Bridge event
    bbox_overlay: List[BBoxOverlayEntry] = []
    boxes = event.get("boxes", [])
    if boxes:
        for i, box in enumerate(boxes):
            if len(box) == 4:
                bx, by, bw, bh = box
                # Estimate time point for overlay entry
                t_point = start + (i * (duration / max(1, len(boxes) - 1))) if len(boxes) > 1 else start
                bbox_overlay.append(
                    BBoxOverlayEntry(
                        time=round(t_point, 2),
                        person_id=str(person_ids[0]) if person_ids else f"person_{track_id or 0}",
                        x=float(bx),
                        y=float(by),
                        w=float(bw),
                        h=float(bh),
                        color=color_tag,
                    )
                )

    # camera_id: resolved via p2_field_resolvers
    camera_id = str(event.get("camera_id") or resolve_camera_id(video_id))

    # exam_phase: resolved via p2_field_resolvers
    v_dur = video_duration if video_duration is not None else float(event.get("video_duration", 0.0))
    exam_phase = str(event.get("exam_phase") or resolve_exam_phase(start, v_dur))

    # repetition_count: resolved via compute_repetition_count if per-frame sequence exists
    if "per_frame_object_detected" in event:
        repetition_count = compute_repetition_count(event["per_frame_object_detected"])
    else:
        repetition_count = int(event.get("repetition_count", 0))

    # TODO: avg_motion_intensity not yet computed by P2P3Bridge
    avg_motion_intensity = float(event.get("avg_motion_intensity", 0.0))

    # TODO: peak_intensity not yet computed by P2P3Bridge
    peak_intensity = float(event.get("peak_intensity", 0.0))

    # TODO: mog2_foreground_ratio not yet computed by P2P3Bridge
    mog2_foreground_ratio = float(event.get("mog2_foreground_ratio", 0.0))

    # TODO: severity_score rule-based calculation pending P3 feature integration
    severity_score = float(event.get("severity_score", 0.8 if object_detected else 0.3))

    # TODO: risk_label pending P3 severity scoring
    risk_label = event.get("risk_label")

    # Rule-based fallback for risk_level/explanation -- the LLM reasoning
    # layer below overrides both ONLY on success; if it's disabled/fails,
    # these rule-based values (risk_level usually None until P3 sets one
    # some other way; explanation a template string) are what ships.
    risk_level = event.get("risk_level")
    explanation = str(event.get("explanation") or f"Activity {event_type} detected on track {track_id or 'unknown'}")

    # TODO: overall event confidence score pending P3 integration
    confidence = float(event.get("confidence", object_confidence if object_confidence is not None else 0.5))

    # thumbnail_path: write a real JPEG if P2P3Bridge captured a frame for
    # this track (see p2_p3_bridge.py's "_thumbnail_frame" internal key);
    # save_event_thumbnail() returns None (not a broken file) if there's
    # nothing to write, so thumbnail_path just stays None as before.
    # One file per EVENT, keyed by this event's start second (Backend's
    # confirmed <videoId>_t<timestamp_start> convention) -- not event_id,
    # since Backend's MongoDB _id doesn't exist yet at write time.
    thumbnail_path = event.get("thumbnail_path") or save_event_thumbnail(
        video_id, event.get("_thumbnail_frame"), timestamp_start=start, event_id=event_id
    )

    # heatmap_ref: same idea, from the accumulated motion-mask stopgap (see
    # thumbnail_heatmap_writer.save_event_heatmap's docstring for why this
    # is a stopgap and not real P1 heatmap output).
    heatmap_ref = event.get("heatmap_ref") or save_event_heatmap(
        video_id, event.get("_heatmap_accum"), timestamp_start=start
    )

    # TODO: related_event_id cross-referencing pending P3/LLM layer
    related_event_id = event.get("related_event_id")

    # LLM reasoning layer (Groq): optional enrichment over the signals just
    # computed above. Only overrides risk_level/explanation, and only on
    # success -- severity_score/risk_label (the rule-based fallback) are
    # NEVER touched here, and a None result (disabled/failed/malformed)
    # leaves risk_level/explanation exactly as the rule-based values already
    # computed above. See reasoning_layer.py for the failure-handling detail.
    llm_result = get_llm_reasoning({
        "seat_id": seat_id,
        "duration": duration,
        "activities": activities,
        "object_detected": object_detected,
        "object_confidence": object_confidence,
        "avg_motion_intensity": avg_motion_intensity,
        "repetition_count": repetition_count,
        "invigilator_excluded": invigilator_excluded,
        "exam_phase": exam_phase,
        "related_event_id": related_event_id,
    })
    if llm_result is not None:
        risk_level = llm_result["risk_level"]
        explanation = llm_result["explanation"]

    return Event(
        event_id=event_id,
        video_id=video_id,
        camera_id=camera_id,
        start=start,
        end=end,
        duration=duration,
        seat_id=seat_id,
        event_type=event_type,
        object_detected=object_detected,
        object_confidence=object_confidence,
        person_ids=person_ids,
        activities=activities,
        avg_motion_intensity=avg_motion_intensity,
        peak_intensity=peak_intensity,
        mog2_foreground_ratio=mog2_foreground_ratio,
        repetition_count=repetition_count,
        invigilator_excluded=invigilator_excluded,
        exam_phase=exam_phase,
        severity_score=severity_score,
        risk_label=risk_label,
        risk_level=risk_level,
        explanation=explanation,
        confidence=confidence,
        color_tag=color_tag,
        thumbnail_path=thumbnail_path,
        bbox_overlay=bbox_overlay,
        heatmap_ref=heatmap_ref,
        related_event_id=related_event_id,
        notes=notes,
    )


def adapt_bridge_events_to_schema(
    events: List[Dict[str, Any]],
    video_id: str,
    event_type: str = DEFAULT_EVENT_TYPE_PLACEHOLDER,
    seat_id: Optional[str] = None,
    notes: Optional[str] = None,
    video_duration: Optional[float] = None,
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
            video_duration=video_duration,
        )
        for e in events
    ]
