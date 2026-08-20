# ============================================================
# segment_events.py
# ============================================================
def segment_events(region_scores_over_time, threshold=30.0, n_on=3, m_off=3):
    """
    Hysteresis segmentation.
    Input: list of {time, seat_id, score}, sorted by time per seat.
    Output: [RawEvent] = {video_id?, seat_id, start_time, end_time, scores:[...]}
    """
    by_seat = {}
    for row in region_scores_over_time:
        by_seat.setdefault(row["seat_id"], []).append(row)

    raw_events = []
    for seat_id, rows in by_seat.items():
        rows.sort(key=lambda r: r["time"])
        on_count, off_count = 0, 0
        in_event = False
        cur_start, cur_scores = None, []

        for row in rows:
            above = row["score"] >= threshold
            if above:
                on_count += 1
                off_count = 0
            else:
                off_count += 1
                on_count = 0

            if not in_event and on_count >= n_on:
                in_event = True
                cur_start = row["time"]
                cur_scores = [row["score"]]
            elif in_event:
                cur_scores.append(row["score"])
                if off_count >= m_off:
                    raw_events.append({
                        "seat_id": seat_id,
                        "start_time": cur_start,
                        "end_time": row["time"],
                        "scores": cur_scores,
                    })
                    in_event = False
                    cur_start, cur_scores = None, []

        if in_event and cur_start is not None:
            raw_events.append({
                "seat_id": seat_id,
                "start_time": cur_start,
                "end_time": rows[-1]["time"],
                "scores": cur_scores,
            })

    return raw_events


if __name__ == "__main__":
    from mock_event_data import generate_region_scores
    scores = generate_region_scores()
    events = segment_events(scores)
    print(f"{len(events)} raw events segmented")
    for e in events[:3]:
        print(e)