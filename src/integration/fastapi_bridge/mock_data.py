import random
import uuid
from schemas import Event, Person


def generate_mock_events(video_id: str, n: int = 5) -> list[Event]:
    events = []
    t = 0.0
    for i in range(n):
        start = t
        end = t + random.uniform(3, 12)
        events.append(Event(
            event_id=str(uuid.uuid4()),
            video_id=video_id,
            start=round(start, 2),
            end=round(end, 2),
            seat_id=f"seat_{random.randint(1, 30)}" if random.random() > 0.5 else None,
            event_type=random.choice(["suspicious_movement", "object_exchange", "seat_swap"]),
            object_detected=random.random() > 0.5,
            object_confidence=round(random.uniform(0.3, 0.95), 2) if random.random() > 0.5 else None,
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
        last_seen=100.0
    )