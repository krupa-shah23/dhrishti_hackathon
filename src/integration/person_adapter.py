"""
Person adapter module — the Re-ID equivalent of event_adapter.py.

Gap this closes: PersonStore (src/track_det/reid.py) holds real, tuned
embeddings keyed by person_id, and p1_p2_tracker.py attaches person_id onto
every fused_track — but nothing ever converts that into a validated
schemas.Person instance, so /internal/persons never actually gets called.
event_adapter.py solved this exact problem for Event; this file does the
same thing for Person, without touching PersonStore's internals.

Known limitation (flagged, not silently hidden): PersonStore.match_or_create()
is only called ONCE per track_id (cached in track_person_map), not once per
frame, so PersonStore has no visibility into how long that person appears —
there is currently no first_seen/last_seen/total_duration signal anywhere in
the pipeline. This adapter fills those three fields with None rather than
fabricating a number. If real duration numbers are needed later, track
{first_frame_ts, last_frame_ts} per track_id in p1_p2_tracker.py's fused_tracks
loop and pass them in via the optional params below — don't derive them here.
"""
from typing import Dict, Any, Optional, List

try:
    from .fastapi_bridge.schemas import Person
except ImportError:
    from src.integration.fastapi_bridge.schemas import Person


def adapt_person_to_schema(
    person_id: str,
    person_data: Dict[str, Any],
    video_id: str,
    seat_id: Optional[str] = None,
    first_seen: Optional[float] = None,
    last_seen: Optional[float] = None,
    total_duration: Optional[float] = None,
) -> Person:
    """
    Converts one entry from PersonStore.persons (keyed by person_id) into a
    validated schemas.Person instance.

    person_data is the dict PersonStore stores per person_id:
        {"embedding": [...], "video_ids": [...], "seat_ids": [...], "thumbnail_path": ...}

    seat_id: pass explicitly if you have the seat for THIS video specifically —
             person_data["seat_ids"] is a list across all videos, not scoped
             to one. Falls back to the first seat_id on record if not given.
    """
    if not video_id:
        raise ValueError("video_id is required to create a valid schemas.Person")

    resolved_seat = seat_id
    if resolved_seat is None and person_data.get("seat_ids"):
        resolved_seat = person_data["seat_ids"][0]

    return Person(
        person_id=person_id,
        video_id=video_id,
        seat_id=resolved_seat,
        first_seen=first_seen,
        last_seen=last_seen,
        embedding_vector=person_data.get("embedding"),
        thumbnail_path=person_data.get("thumbnail_path"),
        total_duration=total_duration,
    )


def adapt_new_or_updated_persons_for_video(
    person_store,  # PersonStore instance
    video_id: str,
    track_person_map: Dict[int, str],
    seat_lookup: Optional[Dict[str, str]] = None,
) -> List[Person]:
    """
    Batch entry point — call this ONCE per video, right after
    pipeline.save_persons(), before cap.release() / video-complete signal.

    track_person_map: the pipeline's own {track_id: person_id} cache
                       (P1P2TrackerPipeline.track_person_map) — used here
                       only to know WHICH person_ids were actually touched
                       in this video, so we don't re-POST every person ever
                       seen across all videos on every single video's completion.
    seat_lookup: optional {person_id: seat_id} for this video specifically,
                 if you have it (e.g. from P1's seat-grid mapping). Falls back
                 to PersonStore's own recorded seat_ids if not provided.

    Returns a list ready to POST individually to /internal/persons — one
    call per person, per the ML doc §9's "once per new/matched person, not
    per frame" rule.
    """
    seen_person_ids = set(track_person_map.values())
    results = []
    for pid in seen_person_ids:
        if pid not in person_store.persons:
            continue  # defensive: shouldn't happen, but don't crash the batch
        data = person_store.persons[pid]
        seat_id = (seat_lookup or {}).get(pid)
        results.append(
            adapt_person_to_schema(
                person_id=pid,
                person_data=data,
                video_id=video_id,
                seat_id=seat_id,
            )
        )
    return results