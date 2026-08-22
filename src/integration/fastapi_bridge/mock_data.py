from __future__ import annotations
import random
import uuid
from schemas import Event, Person, BBoxOverlayEntry


def generate_mock_events(video_id: str, n: int = 5) -> list[Event]:
    events = []
    t = 0.0
    for i in range(n):
        start = t
        end = t + random.uniform(3, 12)
        dur = round(end - start, 2)
        person_id = f"person_{random.randint(1, 10)}"
        event_type = random.choice(["suspicious_movement", "object_exchange", "seat_swap"])
        obj_det = random.random() > 0.5
        obj_conf = round(random.uniform(0.3, 0.95), 2) if obj_det else None
        sev = round(random.uniform(0.2, 0.9), 2)
        color = "red" if sev >= 0.6 else "yellow"

        overlay = [
            BBoxOverlayEntry(
                time=round(start + j * (dur / 3.0), 2),
                person_id=person_id,
                x=round(random.uniform(100, 300), 1),
                y=round(random.uniform(100, 300), 1),
                w=round(random.uniform(50, 120), 1),
                h=round(random.uniform(80, 180), 1),
                color=color,
            )
            for j in range(3)
        ]

        events.append(Event(
            event_id=str(uuid.uuid4()),
            video_id=video_id,
            camera_id=f"cam_{video_id}",
            start=round(start, 2),
            end=round(end, 2),
            duration=dur,
            seat_id=f"seat_{random.randint(1, 30)}" if random.random() > 0.5 else None,
            event_type=event_type,
            object_detected=obj_det,
            object_confidence=obj_conf,
            person_ids=[person_id],
            activities=[event_type],
            avg_motion_intensity=round(random.uniform(10.0, 45.0), 2),
            peak_intensity=round(random.uniform(50.0, 95.0), 2),
            mog2_foreground_ratio=round(random.uniform(0.05, 0.25), 3),
            repetition_count=random.randint(0, 2),
            invigilator_excluded=False,
            exam_phase="main_exam",
            severity_score=sev,
            risk_label="high" if sev >= 0.7 else ("medium" if sev >= 0.4 else "low"),
            risk_level="high" if sev >= 0.7 else ("medium" if sev >= 0.4 else "low"),
            explanation=f"Mock detection: {event_type} observed near seat",
            confidence=round(random.uniform(0.6, 0.98), 2),
            color_tag=color,
            thumbnail_path=None,
            bbox_overlay=overlay,
            heatmap_ref=None,
            related_event_id=None,
            notes="mock event"
        ))
        t = end + random.uniform(1, 5)
    return events


def generate_mock_person(video_id: str) -> Person:
    return Person(
        person_id=str(uuid.uuid4()),
        video_id=video_id,
        seat_id=f"seat_{random.randint(1, 30)}",
        first_seen=0.0,
        last_seen=100.0,
        embedding_vector=[round(random.uniform(-1.0, 1.0), 4) for _ in range(128)],
        thumbnail_path=f"thumbnails/{video_id}_person.jpg",
        total_duration=100.0
    )