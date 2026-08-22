"""
§1: Hysteresis segmentation + incident clustering (P3 joint work)

This module provides:
1. enrich_event_with_motion_fields(): adapter that takes a completed P2P3Bridge event
   and adds the motion-signal fields P3's segment_events/cluster_incidents expect.
2. A thin segment_events() and cluster_incidents() implementation aligned with P3's contract.
3. link_related_events(): links events that share spatial origin and are temporally close.

Signal format P3 expects (verified against P3's API contract):
    avg_motion_intensity    float [0-1]: mean per-seat motion over event duration
    peak_intensity          float [0-1]: max per-seat motion in any single frame
    mog2_foreground_ratio   float [0-1]: fraction of event frames that had >0 foreground
    invigilator_excluded    bool: True if the motion source was identified as the invigilator
    intervention_detected   bool: True if guard check-in fired during this event
    event_type              str:  'phone_use'|'chit_use'|'seat_vacancy'|'talking'|'unknown'

These fields are computed from the p2_p3_bridge event output + optional motion_stats
dict passed in from the frame pipeline. Where motion stats are unavailable (e.g. in
unit tests), safe defaults are used.
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
) -> List[Dict[str, Any]]:
    """
    Applies hysteresis thresholding to filter enriched events into confirmed
    incident segments. Events below motion_threshold or shorter than
    min_duration_sec are discarded as noise.

    Hysteresis: an event must have avg_motion_intensity > motion_threshold
    AND duration >= min_duration_sec to be retained.

    Returns filtered list of events with 'segment_confirmed'=True.
    """
    confirmed = []
    for ev in enriched_events:
        duration = ev.get("end_time", 0.0) - ev.get("start_time", 0.0)
        avg_intensity = ev.get("avg_motion_intensity", 0.0)

        if avg_intensity >= motion_threshold and duration >= min_duration_sec:
            ev["segment_confirmed"] = True
            confirmed.append(ev)

    return confirmed


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

