"""
§1: Hysteresis segmentation + incident clustering (P3 joint work)

This module provides:
1. enrich_event_with_motion_fields(): adapter that takes a completed P2P3Bridge event
   and adds the motion-signal fields P3's segment_events/cluster_incidents expect.
2. A thin segment_events() and cluster_incidents() implementation aligned with P3's contract.
3. link_related_events(): links events that share spatial origin and are temporally close.

Signal format P3 expects (verified against P3's API contract):
    avg_motion_intensity    float [0-1]: mean FUSED (MOG2 OR frame-diff) motion
                             intensity over the event's frames (whole-frame, not
                             literally per-seat -- see §B item 1 in DRISHTI_PA_DONE.md)
    peak_intensity          float [0-1]: max of the same fused signal in any single frame
    mog2_foreground_ratio   float [0-1]: fraction of foreground pixels in MOG2's OWN
                             mask, pre-fusion (distinct signal from the two above)
    invigilator_excluded    bool: True if the motion source was identified as the invigilator
    intervention_detected   bool: True if guard check-in fired during this event
    event_type              str:  'phone_use'|'chit_use'|'seat_vacancy'|'talking'|'unknown'

These fields are computed from the p2_p3_bridge event output + optional motion_stats
dict passed in from the frame pipeline. Where motion stats are unavailable (e.g. in
unit tests), safe defaults are used.

§6 motion-metric contract (Phase 1 lock): as of this session, P2P3Bridge's
finalized event dict exposes genuine per-frame series --
event["motion_intensities"] (fused mask) and event["mog2_ratios"] (MOG2-only
mask, pre-fusion) -- computed from real per-frame measurements
(P1P2TrackerPipeline.process_frame()'s "motion_intensity"/"mog2_foreground_ratio"
return values). A caller can now build a genuine motion_stats dict via
avg()/max() over these lists instead of a placeholder. As of this same
session, scripts/verification/*.py and src/motion/{ablation_study,
benchmark_stages}.py still pass either a hardcoded mog2_foreground_ratio
(1.0 or 0.0, not a real measurement) or omit motion_stats entirely (all three
fields default to 0.0) -- deliberately left unchanged here since rewiring
them would alter already-validated, previously-published results (see
DRISHTI_PA_DONE.md) without independent re-verification. Flagged as a Phase 2
follow-up, not fixed silently.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional


def enrich_event_with_motion_fields(
    event: Dict[str, Any],
    motion_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Adapter: takes a completed P2P3Bridge event dict and adds the motion-signal
    fields that P3's segment_events() and cluster_incidents() consume.

    Parameters
    ----------
    event : dict
        Output from P2P3Bridge._finalize_track(). Must contain:
        object_detected, is_invigilator, metadata[], start_time, end_time.
    motion_stats : dict, optional
        Per-event motion statistics from the live pipeline. Expected keys:
            avg_motion_intensity    float
            peak_intensity          float
            mog2_foreground_ratio   float
            intervention_detected   bool
        If None, safe defaults (0.0 / False) are used.

    Returns
    -------
    dict: event with added fields (mutates and returns in-place).
    """
    stats = motion_stats or {}
    event["avg_motion_intensity"] = float(stats.get("avg_motion_intensity", 0.0))
    event["peak_intensity"] = float(stats.get("peak_intensity", 0.0))
    event["mog2_foreground_ratio"] = float(stats.get("mog2_foreground_ratio", 0.0))
    event["invigilator_excluded"] = bool(event.get("is_invigilator", False))
    event["intervention_detected"] = bool(stats.get("intervention_detected", False))

    # Infer event_type from detection metadata
    classes_detected = [m["class"] for m in event.get("metadata", []) if "class" in m]
    if "phone" in classes_detected:
        event["event_type"] = "phone_use"
    elif "paper-chit" in classes_detected:
        event["event_type"] = "chit_use"
    elif event.get("invigilator_excluded"):
        event["event_type"] = "invigilator_pass"
    else:
        event["event_type"] = "unknown"

    return event


def segment_events(
    enriched_events: List[Dict[str, Any]],
    motion_threshold: float = 0.05,
    min_duration_sec: float = 1.0,
    merge_gap_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Filters and merges enriched events into confirmed incident segments.

    Two-stage process implementing the equivalent of N-on/M-off hysteresis
    for this architecture where events arrive as completed track objects
    (not frame-by-frame signals):

    Stage 1 — N-on gate (open only on confirmed motion):
        Discard events below motion_threshold or shorter than min_duration_sec.
        Also drops invigilator-flagged events (set by enrich_event_with_motion_fields).

    Stage 2 — M-off merge (hold-off before closing):
        Same-seat events separated by a gap < merge_gap_sec are merged into one.
        This prevents one physical action (e.g. a candidate reading) from being
        fragmented into many 1-second track events when the CentroidTracker spawns
        separate IDs for each motion blob during a busy frame window.

        Merge logic:
          - Events are sorted by start_time.
          - Two events A, B are merged if:
              (a) they share at least one seat_id (or both have no seat_ids)
              (b) B.start_time - A.end_time <= merge_gap_sec
          - Merged event inherits: earliest start_time, latest end_time,
            union of seat_ids, highest avg_motion_intensity, first event_id.

    Returns filtered+merged list of events with 'segment_confirmed'=True.
    """
    # ---- Stage 1: N-on gate ----
    confirmed = []
    for ev in enriched_events:
        # Drop invigilator tracks before they generate seat-tagged segments.
        # invigilator_excluded is set by enrich_event_with_motion_fields() from
        # P2P3Bridge's is_invigilator field, which is raised whenever
        # p1_p2_tracker sets invigilator_flag=True on a fused track dict.
        # This is the earliest safe gate: after enrichment (flag is set), before
        # cluster_incidents / link_related_events see the event at all.
        if ev.get("invigilator_excluded", False):
            continue

        duration = ev.get("end_time", 0.0) - ev.get("start_time", 0.0)
        avg_intensity = ev.get("avg_motion_intensity", 0.0)

        if avg_intensity >= motion_threshold and duration >= min_duration_sec:
            ev["segment_confirmed"] = True
            confirmed.append(ev)

    if not confirmed or merge_gap_sec <= 0.0:
        return confirmed

    # ---- Stage 2: M-off merge ----
    # Sort by start_time, then greedily merge seat-adjacent events within merge_gap_sec.
    sorted_evs = sorted(confirmed, key=lambda e: e.get("start_time", 0.0))

    merged = []
    current = dict(sorted_evs[0])  # shallow copy so we don't mutate original
    current["seat_ids"] = list(current.get("seat_ids", []))

    for ev in sorted_evs[1:]:
        seats_cur  = set(current.get("seat_ids", []))
        seats_next = set(ev.get("seat_ids", []))
        gap        = ev.get("start_time", 0.0) - current.get("end_time", 0.0)

        # Merge condition: seat overlap (or both unseated) AND gap within hold-off window
        seats_overlap = bool(seats_cur & seats_next) or (not seats_cur and not seats_next)

        if seats_overlap and gap <= merge_gap_sec:
            # Extend the current merged event
            current["end_time"]             = max(current["end_time"], ev.get("end_time", 0.0))
            current["end_frame"]            = max(current.get("end_frame", 0), ev.get("end_frame", 0))
            current["seat_ids"]             = sorted(seats_cur | seats_next)
            current["avg_motion_intensity"] = max(
                current.get("avg_motion_intensity", 0.0),
                ev.get("avg_motion_intensity", 0.0),
            )
            current["peak_intensity"]       = max(
                current.get("peak_intensity", 0.0),
                ev.get("peak_intensity", 0.0),
            )
            # Extend positions/boxes for centroid calculations downstream
            current["positions"] = current.get("positions", []) + ev.get("positions", [])
            current["boxes"]     = current.get("boxes", []) + ev.get("boxes", [])
        else:
            merged.append(current)
            current = dict(ev)
            current["seat_ids"] = list(current.get("seat_ids", []))

    merged.append(current)
    return merged





def _events_spatially_adjacent(ev_a: Dict[str, Any], ev_b: Dict[str, Any], camera_id: str = "Camera12", adjacency_threshold_px: float = 50.0) -> bool:
    """
    Returns True if two events are in the same or adjacent grid seats based on an explicit
    adjacency map, preventing false overlaps from 2D perspective bounding boxes.
    """
    from src.motion.grid_config import get_adjacent_seats
    
    seats_a = set(ev_a.get("seat_ids", []))
    seats_b = set(ev_b.get("seat_ids", []))
    
    # If no seat data available on either side, don't gate on space
    if not seats_a and not seats_b:
        return True
    
    # If they share any seat directly, they overlap
    if seats_a & seats_b:
        return True
        
    # Check explicit adjacency map
    for s_a in seats_a:
        adj_seats = get_adjacent_seats(s_a, camera_id)
        if seats_b & adj_seats:
            return True
                
    return False


def cluster_incidents(
    confirmed_events: List[Dict[str, Any]],
    time_gap_sec: float = 5.0,
    camera_id: str = "Camera12",
) -> List[List[Dict[str, Any]]]:
    """
    Groups temporally AND spatially adjacent confirmed events into incident clusters.

    Two events are merged into the same cluster only if:
      1. They are temporally close: start_time gap from previous event's end_time <= time_gap_sec
      2. They are spatially adjacent: they share at least one seat_id (same/adjacent seat region)

    Pure time-only clustering was collapsing all events across the room into one giant cluster
    on clips with continuous motion. The spatial gate ensures events in different parts of the
    room start separate clusters even when temporally close.

    Returns: list of clusters, where each cluster is a list of events.
    """
    if not confirmed_events:
        return []

    # Sort by start_time
    sorted_events = sorted(confirmed_events, key=lambda e: e.get("start_time", 0.0))

    clusters = []
    current_cluster = [sorted_events[0]]

    for ev in sorted_events[1:]:
        prev_ev = current_cluster[-1]
        prev_end = prev_ev.get("end_time", 0.0)
        curr_start = ev.get("start_time", 0.0)

        time_close = (curr_start - prev_end) <= time_gap_sec
        spatially_adjacent = _events_spatially_adjacent(prev_ev, ev, camera_id=camera_id)

        if time_close and spatially_adjacent:
            current_cluster.append(ev)
        else:
            clusters.append(current_cluster)
            current_cluster = [ev]

    clusters.append(current_cluster)
    return clusters


def link_related_events(
    clusters: List[List[Dict[str, Any]]],
    spatial_radius_px: float = 150.0,  # kept for API compat, not used when seat_ids present
    time_window_sec: float = 60.0,
    camera_id: str = "Camera12",
) -> List[Dict[str, Any]]:
    """
    Links events that share spatial origin (same/adjacent seat_ids, or centroid fallback)
    and occur within time_window_sec of each other across separate clusters.
    Sets 'related_event_id' on each event that has a related counterpart.

    Events in the SAME cluster are not linked (they're already merged).
    Events in DIFFERENT clusters that are spatially+temporally close are linked.

    Spatial proximity: if seat_ids are present on both events, uses seat_id overlap
    (any shared seat). Falls back to centroid pixel distance (spatial_radius_px) if
    seat_ids are absent (backward compatibility).

    Returns flat list of all events with related_event_id populated where applicable.
    """
    import math

    def centroid(ev):
        positions = ev.get("positions", [])
        if not positions:
            return None
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        return (cx, cy)

    def dist(c1, c2):
        if c1 is None or c2 is None:
            return float("inf")
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def spatially_related(ev_a, ev_b):
        """True if events are in adjacent seat regions geographically, or centroid-close as fallback."""
        seats_a = set(ev_a.get("seat_ids", []))
        seats_b = set(ev_b.get("seat_ids", []))
        if seats_a and seats_b:
            # Use the physical grid-adjacency function
            return _events_spatially_adjacent(ev_a, ev_b, camera_id=camera_id, adjacency_threshold_px=50.0)
        # Fallback: centroid pixel distance
        return dist(centroid(ev_a), centroid(ev_b)) <= spatial_radius_px

    # Flatten all events, tagging with cluster index
    flat_events = []
    for ci, cluster in enumerate(clusters):
        for ev in cluster:
            ev["_cluster_idx"] = ci
            flat_events.append(ev)

    # Pairwise link across different clusters
    for i, ev_a in enumerate(flat_events):
        for j, ev_b in enumerate(flat_events):
            if i >= j:
                continue
            if ev_a["_cluster_idx"] == ev_b["_cluster_idx"]:
                continue  # Same cluster, not cross-linked

            time_delta = abs(ev_a.get("start_time", 0.0) - ev_b.get("start_time", 0.0))
            if time_delta > time_window_sec:
                continue

            if spatially_related(ev_a, ev_b):
                # Link bidirectionally
                ev_a["related_event_id"] = ev_b.get("event_id")
                ev_b["related_event_id"] = ev_a.get("event_id")

    # Remove internal cluster index tag
    for ev in flat_events:
        ev.pop("_cluster_idx", None)

    return flat_events

