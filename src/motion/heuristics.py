import numpy as np
from typing import Any, Dict

from .grid_config import get_adjacent_seats

# Maximum number of consecutive frames with >2 simultaneous active cells
# that is still treated as brief noise. A sustained pattern beyond this
# threshold is treated as a single moving body (e.g., a nearby guard whose
# silhouette spans multiple grid cells due to perspective geometry).
_BRIEF_SPIKE_MAX_FRAMES = 3

# §9 build_explanation: maps a raw `activities[]` signal string (from
# pose_gesture.py / this module) to the human-readable phrase used in the
# explanation template. `chit_passing` carries a "chit_passing:<tid_a>+<tid_b>"
# suffix, so it's matched by prefix, not exact string, below.
_ACTIVITY_DESCRIPTIONS = {
    "sustained_gaze_shift": "sustained gaze shift",
    "targeted_scanning": "targeted scanning",
    "chit_passing": "hand movement toward neighbor",
    "seat_vacant_near_invigilator": "seat vacant near invigilator",
}
_NO_ACTIVITY_DESCRIPTION = "motion detected"


def is_invigilator_motion(cell_history: list[set[str]]) -> bool:
    """
    Detects a multi-cell traveling pattern (invigilator walking past).

    cell_history: Sequence of sets of active grid-cell IDs over a short window.

    Returns True if:
    1. Any simultaneous multi-cell spike (>2 cells active in one frame) is
       either absent OR sustained (> _BRIEF_SPIKE_MAX_FRAMES consecutive frames).
       Brief spikes (1-3 frames) are rejected as noise (e.g. candidates shifting
       at once).  Sustained wide-activation is treated as a single moving body.
    2. The chronological sequence of active cells transitions across at least
       3 distinct cells.

    Design note (§0A fix):
       The original hard-reject `any(len(active) > 2)` false-rejects a guard
       walking near-camera on DAHISAR1 where perspective geometry causes a
       single body to span 5-7 overlapping grid cells simultaneously for the
       full duration of the walk. The persistence check distinguishes:
         - Brief (≤3 frames) wide spike  → noise, reject
         - Sustained (>3 frames) wide spike → single moving source, allow
    """
    if not cell_history:
        return False

    # --- Persistence-aware simultaneous-spike gate ---
    # Count the longest consecutive run of frames with >2 active cells.
    max_run = 0
    current_run = 0
    for active in cell_history:
        if len(active) > 2:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    if max_run > 0 and max_run <= _BRIEF_SPIKE_MAX_FRAMES:
        # Brief spike only — this is noise (e.g., whole room shifts together), reject.
        return False
    # If max_run == 0: no wide frames at all — proceed normally.
    # If max_run > _BRIEF_SPIKE_MAX_FRAMES: sustained single-body motion — proceed.

    # --- Sequential distinct-cell check ---
    active_sequence = []
    for active_cells in cell_history:
        if len(active_cells) == 1:
            cell = list(active_cells)[0]
            if not active_sequence or active_sequence[-1] != cell:
                active_sequence.append(cell)
        elif len(active_cells) == 2:
            # For 2 active cells, if one is new, we consider it a transition step.
            cells = list(active_cells)
            if not active_sequence:
                active_sequence.append(cells[0])
            else:
                for cell in cells:
                    if cell != active_sequence[-1]:
                        active_sequence.append(cell)
                        break
        elif len(active_cells) > 2:
            # Sustained wide-activation frame: treat the whole set as a step.
            # Any cell not already at the end of the sequence counts as a new visit.
            for cell in sorted(active_cells):
                if not active_sequence or active_sequence[-1] != cell:
                    active_sequence.append(cell)
                    break  # one new cell per frame is enough

    distinct_cells = set(active_sequence)
    # Require at least 3 distinct cells visited sequentially
    if len(distinct_cells) >= 3 and len(active_sequence) >= 3:
        return True

    return False

def tag_interventions(events: list[dict], full_cell_history: list[tuple[float, set[str]]]) -> None:
    """
    Tags interventions in-place on flagged events.
    
    events: list of dicts, each with 'start_sec', 'end_sec', 'intervention_detected' (bool)
    full_cell_history: list of (timestamp_sec, active_seat_ids)
    
    If `is_invigilator_motion` fires within/near the active event window using the restricted cell history,
    intervention_detected is set to True.
    """
    for event in events:
        start_sec = event.get('start_sec', 0.0)
        end_sec = event.get('end_sec', 0.0)
        # Pad window slightly to catch the invigilator walking up/away
        window_start = start_sec - 2.0
        window_end = end_sec + 2.0
        
        # Restrict history to event window
        window_history = [
            active_seats for t, active_seats in full_cell_history 
            if window_start <= t <= window_end
        ]
        
        if is_invigilator_motion(window_history):
            event['intervention_detected'] = True
        else:
            event['intervention_detected'] = False

def is_seat_vacated(seat_history: list[bool], timeout_frames: int) -> bool:
    """
    Detects sustained absence of motion/presence beyond a timeout threshold.
    
    seat_history: sequence of boolean presence/motion states (True=active, False=inactive)
    timeout_frames: strictly greater than this number of consecutive False values triggers vacancy.
    """
    if not seat_history:
        return False
        
    consecutive_inactive = 0
    for state in reversed(seat_history):
        if not state:
            consecutive_inactive += 1
        else:
            break
            
    # Strictly greater than timeout_frames
    return consecutive_inactive > timeout_frames


def flag_seat_vacant_near_invigilator(seat_history: list[bool], invigilator_cell_history: list[set[str]],
                                       seat_id: str, camera_id: str) -> bool:
    """
    §3.4 seat_vacant_near_invigilator: a student's motion drops out (is_seat_vacated) in
    the same window an invigilator is detected traveling through the
    student's own seat or an adjacent one (is_invigilator_motion).

    seat_history: sequence of boolean presence/motion states for `seat_id`
        over the same window as invigilator_cell_history (see is_seat_vacated).
    invigilator_cell_history: sequence of sets of active grid-cell IDs over
        the same window (see is_invigilator_motion).
    seat_id, camera_id: identify the student's seat and which camera's
        adjacency map to check (see grid_config.get_adjacent_seats).

    Returns True only if BOTH gates fire:
      1. is_invigilator_motion(invigilator_cell_history) is True, AND the
         cells touched anywhere in that window include seat_id itself or a
         seat adjacent to it.
      2. is_seat_vacated(seat_history, timeout_frames=3) is True.
    timeout_frames=3 reuses _BRIEF_SPIKE_MAX_FRAMES, the same noise-vs-signal
    cutoff already used inside is_invigilator_motion.
    Cameras with no adjacency map (get_adjacent_seats returns an empty set)
    degrade Gate 1 to a same-seat-only check.
    """
    if not is_invigilator_motion(invigilator_cell_history):
        return False

    touched_cells: set[str] = set()
    for active in invigilator_cell_history:
        touched_cells |= active

    relevant_cells = {seat_id} | get_adjacent_seats(seat_id, camera_id)
    if not (touched_cells & relevant_cells):
        return False

    return is_seat_vacated(seat_history, timeout_frames=_BRIEF_SPIKE_MAX_FRAMES)


def _activity_description(activities: list) -> str:
    """
    Maps the first entry of an event's `activities[]` list to a human-readable
    phrase via `_ACTIVITY_DESCRIPTIONS` (prefix-matched, since chit_passing
    carries a ":<tid_a>+<tid_b>" suffix). Falls back to the raw signal name
    (underscores -> spaces) for any future signal not yet in the map, and to
    `_NO_ACTIVITY_DESCRIPTION` when the list is empty (motion+fusion only,
    no pose/gesture signal fired) -- this is the case build_explanation()
    must still handle cleanly, never leaving the explanation blank.
    """
    if not activities:
        return _NO_ACTIVITY_DESCRIPTION

    first = activities[0]
    for prefix, description in _ACTIVITY_DESCRIPTIONS.items():
        if first == prefix or first.startswith(prefix + ":"):
            return description

    return first.replace("_", " ")


def build_explanation(event: Dict[str, Any]) -> str:
    """
    §9 integration contract: builds the human-readable `explanation` string
    (ML master doc §3.7 format) from fields already present on a finalized
    event -- same wiring point as severity_score/risk_label/color_tag/confidence
    in p2_p3_bridge.py's _finalize_track().

    Template: "Seat {seat_id}, {activity description}, {duration}s{, {class} detected {confidence}}"
    e.g. "Seat 14, hand movement toward neighbor, 6.4s, phone detected 0.62"

    Deterministic string formatting only -- no NLP, no randomness.

    Parameters
    ----------
    event : dict
        A finalized event as produced by P2P3Bridge._finalize_track(). Reads
        (all optional/defensive -- never raises on a missing key):
            seat_ids          list[str]
            activities        list[str]
            start_time, end_time  float (seconds)
            object_detected   bool
            metadata          list[dict] with 'class'/'confidence' keys

    Returns
    -------
    str: never blank -- motion+fusion alone (no seat, no activities, no
    object) still produces a valid explanation via the "unknown seat" /
    "motion detected" fallbacks.
    """
    # grid_config.py's real seat_ids are "seat_1", "seat_66", etc. -- strip the
    # "seat_" prefix so the template's own "Seat " word isn't duplicated
    # (matches the master-doc example "Seat 14", not "Seat seat_14").
    seat_ids = event.get("seat_ids") or []
    seat_numbers = [s[len("seat_"):] if s.startswith("seat_") else s for s in seat_ids]
    seat_label = "/".join(seat_numbers) if seat_numbers else "unknown"

    activity_desc = _activity_description(event.get("activities") or [])

    duration = float(event.get("end_time", 0.0)) - float(event.get("start_time", 0.0))
    duration = max(0.0, duration)

    object_clause = ""
    if event.get("object_detected"):
        metadata = event.get("metadata") or []
        dets = [m for m in metadata if m.get("class")]
        if dets:
            best = max(dets, key=lambda m: m.get("confidence") if m.get("confidence") is not None else -1.0)
            cls = best["class"]
            conf = best.get("confidence")
            if conf is not None:
                object_clause = f", {cls} detected {conf:.2f}"
            else:
                object_clause = f", {cls} detected"

    return f"Seat {seat_label}, {activity_desc}, {duration:.1f}s{object_clause}"
