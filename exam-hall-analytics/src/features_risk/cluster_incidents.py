# ============================================================
# cluster_incidents.py
# ============================================================
def _adjacent(seat_a, seat_b, max_desk_gap=1):
    """Simple adjacency: same seat, or numeric desk id within gap."""
    if seat_a == seat_b:
        return True
    try:
        na = int(seat_a.replace("Desk", ""))
        nb = int(seat_b.replace("Desk", ""))
        return abs(na - nb) <= max_desk_gap
    except ValueError:
        return False


def cluster_incidents(raw_events, time_gap=5.0, max_desk_gap=0):
    """
    Merge RawEvents into Events if spatially adjacent AND time-close.
    time_gap: max seconds between end_time of one and start_time of next to merge.
    """
    events = sorted(raw_events, key=lambda e: e["start_time"])
    clustered = []
    used = [False] * len(events)

    for i, e in enumerate(events):
        if used[i]:
            continue
        group = [e]
        used[i] = True
        for j in range(i + 1, len(events)):
            if used[j]:
                continue
            cand = events[j]
            last = group[-1]
            close_in_time = (cand["start_time"] - last["end_time"]) <= time_gap
            close_in_space = _adjacent(last["seat_id"], cand["seat_id"], max_desk_gap)
            if close_in_time and close_in_space:
                group.append(cand)
                used[j] = True

        merged = {
            "event_id": None,  # assign later
            "seat_id": group[0]["seat_id"],
            "start_time": group[0]["start_time"],
            "end_time": group[-1]["end_time"],
            "grid_cells": [g["seat_id"] for g in group],
            "sub_event_count": len(group),
            "scores": [s for g in group for s in g.get("scores", [])],
        }
        clustered.append(merged)

    for idx, ev in enumerate(clustered):
        ev["event_id"] = f"C{idx+1:04d}"

    return clustered


if __name__ == "__main__":
    from mock_event_data import generate_region_scores
    from segment_events import segment_events

    scores = generate_region_scores(seat_ids=["Desk4"], duration_sec=200)
    raw = segment_events(scores, threshold=25, n_on=2, m_off=2)
    clustered = cluster_incidents(raw, time_gap=20)
    print(f"{len(raw)} raw -> {len(clustered)} clustered")
    for c in clustered:
        print(c)