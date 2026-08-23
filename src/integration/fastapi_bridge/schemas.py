from pydantic import BaseModel
from typing import List, Optional, Tuple


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
    # Simplified stand-in for the original bbox_overlay spec
    # ({time, person_id, x, y, w, h, color}) — not implemented under time
    # pressure. Carries the raw per-track/per-frame (x, y, w, h) tuples as
    # produced by P2P3Bridge's event["boxes"] (p2_p3_bridge.py:77) — NOT
    # (x1,y1,x2,y2); corrected after the real-Node smoke test caught a
    # negative-width bug from an earlier wrong assumption here.
    boxes: List[Tuple[float, float, float, float]] = []


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