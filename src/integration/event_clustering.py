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

    if "avg_motion_intensity" in stats and "peak_intensity" in stats:
        avg_int = float(stats.get("avg_motion_intensity", 0.0))
        peak_int = float(stats.get("peak_intensity", 0.0))
    elif "motion_intensities" in event and event["motion_intensities"]:
        avg_int = sum(event["motion_intensities"]) / len(event["motion_intensities"])
        peak_int = max(event["motion_intensities"])
    else:
        avg_int = float(stats.get("avg_motion_intensity", 0.0))
        peak_int = float(stats.get("peak_intensity", 0.0))

    if "mog2_foreground_ratio" in stats:
        mog2_ratio = float(stats.get("mog2_foreground_ratio", 0.0))
    elif "mog2_ratios" in event and event["mog2_ratios"]:
        mog2_ratio = sum(event["mog2_ratios"]) / len(event["mog2_ratios"])
    else:
        mog2_ratio = float(stats.get("mog2_foreground_ratio", 0.0))

    event["avg_motion_intensity"] = avg_int
    event["peak_intensity"] = peak_int
    event["mog2_foreground_ratio"] = mog2_ratio
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


def _reconstruct_frame_timestamps(event: Dict[str, Any]) -> Optional[List[float]]:
    """
    Maps event['frame_indices'] to real seconds using the event's own
    start_frame/start_time/end_frame/end_time -- avoids needing fps threaded
    through segment_events() separately, since P2P3Bridge already computed
    those from its own self.fps when it finalized the event.
    """
    frame_indices = event.get("frame_indices")
    start_frame, end_frame = event.get("start_frame"), event.get("end_frame")
    start_time, end_time = event.get("start_time"), event.get("end_time")
    if not frame_indices or None in (start_frame, end_frame, start_time, end_time):
        return None

    frame_span = end_frame - start_frame
    if frame_span <= 0:
        return [start_time for _ in frame_indices]

    sec_per_frame_unit = (end_time - start_time) / frame_span
    return [start_time + (fi - start_frame) * sec_per_frame_unit for fi in frame_indices]


def _cap_run_duration(s: int, e: int, timestamps: List[float], max_duration_sec: float) -> List[List[int]]:
    """
    Force-splits a single run [s, e] into consecutive chunks of at most
    max_duration_sec each, regardless of intensity. Used only as a pragmatic
    fallback for runs the intensity-based hysteresis above could never close
    on its own (see max_duration_sec's docstring note in segment_events()).
    """
    if timestamps[e] - timestamps[s] <= max_duration_sec:
        return [[s, e]]
    chunks = []
    chunk_start = s
    for i in range(s, e + 1):
        if timestamps[i] - timestamps[chunk_start] >= max_duration_sec:
            chunks.append([chunk_start, i - 1])
            chunk_start = i
    chunks.append([chunk_start, e])
    return chunks


def _split_event_by_motion_intensity(
    event: Dict[str, Any],
    motion_threshold: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    max_duration_sec: float,
) -> List[Dict[str, Any]]:
    """
    Splits ONE finalized P2P3Bridge event (a single continuously-tracked
    track_id's whole lifetime) into ground-truth-scale activity bursts, using
    the event's own per-frame fused motion_intensities series -- instead of
    gating the track's entire lifetime as one atomic pass/fail span.

    Root cause this addresses: P2P3Bridge only finalizes a track when it goes
    missing for missing_threshold frames (or at flush()). A continuously-
    tracked, mostly-seated subject never goes "missing," so its whole
    lifetime -- which can span an entire clip -- previously arrived here as
    ONE event, gated only by its clip-duration-average intensity. Short
    ground-truth-scale bursts (a few seconds) inside a 100+ second presence
    were invisible to that gate: either the average was high enough and the
    ENTIRE span survived as one giant "event" (near-zero IoU against any
    short ground-truth window), or it wasn't and the whole thing was dropped.

    This performs real N-on/M-off hysteresis on the intensity trace INSIDE
    that one event: build runs of consecutive frames at/above
    motion_threshold, bridge runs separated by a gap <= merge_gap_sec (the
    same "hold off before closing" semantics Stage 2 already uses across
    separate events -- just applied within one), then keep only bridged runs
    lasting >= min_duration_sec, each becoming its own sub-event.

    KNOWN LIMITATION, not a final fix: on real clips checked so far, the
    whole-frame fused motion_intensity trace for a continuously-seated
    subject never actually dips back below motion_threshold once MOG2's
    warmup ramp settles -- it plateaus at an elevated, noisy level for the
    rest of the clip regardless of the labeled behavior underneath it. Real
    N-on/M-off hysteresis alone therefore still yields ONE run spanning the
    whole track on that data; there is no genuine "return to baseline" for
    it to find. max_duration_sec is a pragmatic, explicitly-not-a-real-fix
    escape hatch for exactly that case: any run (post-bridging) longer than
    max_duration_sec is force-cut into fixed-length chunks regardless of
    intensity, so a continuously-active track still yields multiple
    bounded-duration events instead of one clip-spanning blob. This does NOT
    locate real activity-burst boundaries -- it only bounds the damage a
    non-discriminating intensity signal can do to event granularity. Actual
    boundary accuracy would need a per-seat/per-ROI intensity signal (not
    whole-frame) or a relative/adaptive threshold -- out of scope here.

    Falls back to returning the event unchanged when per-frame data isn't
    available (e.g. minimal synthetic test events that set
    avg_motion_intensity directly without motion_intensities/frame_indices),
    so every existing caller that never had this data keeps its old behavior.
    """
    intensities = event.get("motion_intensities")
    frame_indices = event.get("frame_indices")
    timestamps = _reconstruct_frame_timestamps(event)

    if not intensities or not frame_indices or not timestamps or len(intensities) != len(frame_indices):
        return [event]

    # ---- Raw on/off runs from the per-frame trace ----
    runs = []  # [start_idx, end_idx] into intensities/timestamps/frame_indices, inclusive
    active_start = None
    for i, val in enumerate(intensities):
        if val >= motion_threshold:
            if active_start is None:
                active_start = i
        elif active_start is not None:
            runs.append([active_start, i - 1])
            active_start = None
    if active_start is not None:
        runs.append([active_start, len(intensities) - 1])

    if not runs:
        return []  # never crossed the intensity gate at all

    # ---- Bridge runs separated by a gap <= merge_gap_sec ----
    merged_runs = [runs[0]]
    for run in runs[1:]:
        prev = merged_runs[-1]
        if timestamps[run[0]] - timestamps[prev[1]] <= merge_gap_sec:
            prev[1] = run[1]
        else:
            merged_runs.append(run)

    # ---- Pragmatic cap: force-cut any bridged run longer than max_duration_sec ----
    # Must happen AFTER bridging, not before -- capping first would just
    # produce back-to-back (zero-gap) chunks that bridging would immediately
    # re-merge, undoing the cap.
    if max_duration_sec > 0:
        capped_runs = []
        for (s, e) in merged_runs:
            capped_runs.extend(_cap_run_duration(s, e, timestamps, max_duration_sec))
        merged_runs = capped_runs

    # ---- Keep only bridged/capped runs that meet min_duration_sec ----
    qualifying_runs = [
        (s, e) for (s, e) in merged_runs
        if timestamps[e] - timestamps[s] >= min_duration_sec
    ]

    # Only append a "_segN" suffix when a genuine split into multiple pieces
    # happened. A single qualifying run keeps the parent's own event_id --
    # matters for callers (e.g. event_id-based assertions, idempotency by
    # event_id) that expect an unsplit event's identity to pass through
    # unchanged when there was nothing to split.
    multi = len(qualifying_runs) > 1

    sub_events = []
    for run_idx, (s, e) in enumerate(qualifying_runs):
        seg_intensities = intensities[s:e + 1]
        sub = dict(event)
        if multi:
            sub["event_id"] = f"{event.get('event_id', 'event')}_seg{run_idx}"
        sub["start_time"], sub["end_time"] = timestamps[s], timestamps[e]
        sub["start_frame"], sub["end_frame"] = frame_indices[s], frame_indices[e]
        sub["total_frames"] = frame_indices[e] - frame_indices[s] + 1
        sub["avg_motion_intensity"] = float(sum(seg_intensities) / len(seg_intensities))
        sub["peak_intensity"] = float(max(seg_intensities))
        sub["motion_intensities"] = seg_intensities
        sub["frame_indices"] = frame_indices[s:e + 1]
        # Trim positions/boxes to this sub-range too, so downstream consumers
        # (e.g. link_related_events' centroid fallback) reflect this burst,
        # not the whole original track's lifetime.
        if event.get("positions") and len(event["positions"]) == len(frame_indices):
            sub["positions"] = event["positions"][s:e + 1]
        if event.get("boxes") and len(event["boxes"]) == len(frame_indices):
            sub["boxes"] = event["boxes"][s:e + 1]
        sub_events.append(sub)

    return sub_events


def segment_events(
    enriched_events: List[Dict[str, Any]],
    motion_threshold: float = 0.05,
    min_duration_sec: float = 1.0,
    merge_gap_sec: float = 2.0,
    max_duration_sec: float = 45.0,
) -> List[Dict[str, Any]]:
    """
    Filters and merges enriched events into confirmed incident segments.

    Three-stage process implementing the equivalent of N-on/M-off hysteresis
    for this architecture where events arrive as completed track objects
    (not frame-by-frame signals):

    Stage 1 — N-on gate (open only on confirmed motion):
        Splits each event into intra-event activity bursts using its own
        per-frame motion_intensities trace (see _split_event_by_motion_intensity),
        then discards bursts below motion_threshold or shorter than
        min_duration_sec. Also drops invigilator-flagged events (set by
        enrich_event_with_motion_fields) before splitting.

        max_duration_sec is a pragmatic cap, NOT real activity-boundary
        detection: on real clips, a continuously-seated subject's whole-frame
        fused motion_intensity was found to never actually return below
        motion_threshold once MOG2's warmup settles (it plateaus at an
        elevated, noisy level for the rest of the clip). Real intensity-based
        hysteresis alone therefore still produces one clip-spanning event on
        that data. max_duration_sec forces any such run to be cut into
        fixed-length chunks after bridging, so a continuously-tracked
        presence still yields multiple bounded events instead of one giant
        blob -- it does not locate genuine behavior-change boundaries. Set
        <= 0 to disable (restores unbounded event duration). Fixing this
        properly needs a per-seat/per-ROI intensity signal or an
        adaptive/relative threshold, not attempted here.

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

        # Split into intra-event activity bursts first (see
        # _split_event_by_motion_intensity's docstring for why: a single
        # continuously-tracked track's whole lifetime otherwise arrives here
        # as one atomic span, hiding ground-truth-scale bursts inside it).
        # Falls back to [ev] unchanged when per-frame data isn't available.
        for seg in _split_event_by_motion_intensity(ev, motion_threshold, min_duration_sec, merge_gap_sec, max_duration_sec):
            duration = seg.get("end_time", 0.0) - seg.get("start_time", 0.0)
            avg_intensity = seg.get("avg_motion_intensity", 0.0)

            if avg_intensity >= motion_threshold and duration >= min_duration_sec:
                seg["segment_confirmed"] = True
                confirmed.append(seg)

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

