from pydantic import BaseModel
from typing import Optional


class Event(BaseModel):
    event_id: str
    video_id: str
    start: float
    end: float
    seat_id: Optional[str] = None
    event_type: str
    object_detected: bool
    object_confidence: Optional[float] = None
    notes: Optional[str] = None


class Person(BaseModel):
    person_id: str
    video_id: str
    seat_id: Optional[str] = None
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None


class CompleteSignal(BaseModel):
    video_id: str
    status: str = "done"
    total_events: Optional[int] = None