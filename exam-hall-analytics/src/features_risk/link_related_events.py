# ============================================================
# link_related_events.py
# ============================================================
def link_related_events(events, time_gap_min=20, time_gap_max=90, max_desk_gap=0):
    """
    Sets related_event_id on events that are same/adjacent seat, gap
    between time_gap_min and time_gap_max seconds (too far apart to
    cluster into one incident, but clearly connected — e.g. clip 4's
    two talking bursts ~45-70s apart).
    """
    def adjacent(a, b):
        if a == b:
            return True
        try:
            na = int(a.replace("Desk", ""))
            nb = int(b.replace("Desk", ""))
            return abs(na - nb) <= max_desk_gap
        except ValueError:
            return False

    evs = sorted(events, key=lambda e: e["start_time"])
    for i, e in enumerate(evs):
        for j in range(i + 1, len(evs)):
            other = evs[j]
            gap = other["start_time"] - e["end_time"]
            if gap > time_gap_max:
                break
            if gap >= time_gap_min and adjacent(e["seat_id"], other["seat_id"]):
                e["related_event_id"] = other.get("event_id")
                other["related_event_id"] = e.get("event_id")
    return evs


if __name__ == "__main__":
    mock = [
        {"event_id": "C0001", "seat_id": "Desk4", "start_time": 10, "end_time": 15},
        {"event_id": "C0002", "seat_id": "Desk4", "start_time": 60, "end_time": 66},
    ]
    linked = link_related_events(mock)
    for e in linked:
        print(e["event_id"], "->", e.get("related_event_id"))