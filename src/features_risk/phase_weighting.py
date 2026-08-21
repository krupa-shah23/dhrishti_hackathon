def get_exam_phase(start_time: float, video_duration: float) -> str:
    """Returns 'start' | 'mid' | 'end' based on % of video duration."""
    if video_duration <= 0:
        return "mid"
    pct = start_time / video_duration
    if pct <= 0.10:
        return "start"
    elif pct >= 0.90:
        return "end"
    return "mid"

def phase_weight_multiplier(phase: str) -> float:
    """Higher multiplier = higher threshold = less sensitive (per v3 §1)."""
    return {"start": 2.0, "mid": 1.0, "end": 2.0}.get(phase, 1.0)

def apply_phase_weighting(event: dict, video_duration: float) -> dict:
    phase = get_exam_phase(event["start_time"], video_duration)
    event["exam_phase"] = phase
    weight = phase_weight_multiplier(phase)
    event["avg_motion_intensity"] = event.get("avg_motion_intensity", 0) / weight
    return event

if __name__ == "__main__":
    from mock_event_data import generate_mock_events
    events = generate_mock_events(5)
    for e in events:
        print(e["event_id"], apply_phase_weighting(e, video_duration=143.12))