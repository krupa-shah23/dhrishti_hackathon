import random

def generate_mock_events(n=20, seed=42, video_id="mock_clip_01"):
    random.seed(seed)
    events = []
    for i in range(n):
        start = round(random.uniform(0, 500), 2)
        duration = round(random.uniform(0.5, 8.0), 2)
        obj = random.choice([None, "phone", "paper"])
        events.append({
            "event_id": f"E{i+1:04d}",
            "video_id": video_id,
            "camera_id": "cam1",
            "seat_id": f"Desk{random.randint(1,20)}",
            "start_time": start,
            "end_time": start + duration,
            "grid_cells": [random.randint(0,50)],
            "avg_motion_intensity": round(random.uniform(0,100),2),
            "peak_intensity": round(random.uniform(0,150),2),
            "mog2_foreground_ratio": round(random.uniform(0,1),3),
            "duration": duration,
            "repetition_count": random.randint(1,4),
            "object_detected": obj,
            "object_confidence": round(random.uniform(0.5,0.95),2) if obj else None,
            "invigilator_excluded": False,
            "exam_phase": random.choice(["start","mid","end"]),
            "exam_mode": random.choice(["CBT", "paper_pen"]),
            "severity_score": None,
            "risk_label": None,
            "explanation": None,
            "audio_corroborated": random.choice([True, False, None]),
            "intervention_detected": random.random() < 0.1,
            "related_event_id": None,
            "event_type": random.choice(["anomaly","seat_vacated","invigilator_absent"]) if random.random() < 0.15 else "anomaly",
        })
    return events


def generate_region_scores(seat_ids=None, duration_sec=500, step=1.0, seed=42):
    random.seed(seed)
    seat_ids = seat_ids or [f"Desk{i}" for i in range(1, 6)]
    scores = []
    active_spike = {s: 0 for s in seat_ids}  # remaining spike ticks per seat

    t = 0.0
    while t < duration_sec:
        for s in seat_ids:
            if active_spike[s] > 0:
                val = random.uniform(30, 60)
                active_spike[s] -= 1
            else:
                val = random.uniform(0, 15)
                if random.random() < 0.02:  # start a new spike
                    active_spike[s] = random.randint(3, 6)  # lasts 3-6 ticks
            scores.append({"time": round(t, 2), "seat_id": s, "score": round(val, 2)})
        t += step
    return scores


def generate_flow_history(sustained=True, length=15, seed=42):
    """Mock optical-flow direction history for glance_persistence testing."""
    random.seed(seed)
    if sustained:
        return [{"time": i, "direction": 45.0 + random.uniform(-3, 3)} for i in range(length)]
    return [{"time": i, "direction": random.uniform(0, 360)} for i in range(length)]


if __name__ == "__main__":
    events = generate_mock_events(5)
    for e in events:
        print(e["event_id"], e["seat_id"], e["duration"], e["event_type"])