import numpy as np

# Maximum number of consecutive frames with >2 simultaneous active cells
# that is still treated as brief noise. A sustained pattern beyond this
# threshold is treated as a single moving body (e.g., a nearby guard whose
# silhouette spans multiple grid cells due to perspective geometry).
_BRIEF_SPIKE_MAX_FRAMES = 3


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
