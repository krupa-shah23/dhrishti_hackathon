from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class BBoxOverlayEntry(BaseModel):
    time: float
    person_id: str
    x: float
    y: float
    w: float
    h: float
    color: str


class Event(BaseModel):
    # --- existing thin-version fields, kept as-is ---
    event_id: str
    video_id: str
    start: float
    end: float
    seat_id: Optional[str] = None
    event_type: Optional[str] = None
    notes: Optional[str] = None

    # --- P2-owned, locked shape (do not change) ---
    object_detected: bool
    object_confidence: Optional[float] = None

    # --- full contract expansion (ML doc §9A / consolidated todo §2.5) ---
    camera_id: str
    person_ids: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    avg_motion_intensity: float
    peak_intensity: float
    mog2_foreground_ratio: float
    duration: float
    repetition_count: int = 0
    invigilator_excluded: bool = False
    exam_phase: str

    # --- severity / explainability ---
    severity_score: float           # rule-based, kept as fallback per your LLM-layer doc
    risk_label: Optional[str] = None    # rule-based color/tier label (yellow/red-adjacent)
    risk_level: Optional[str] = None    # NEW: LLM-reasoned low/medium/high (see open Q2 above)
    explanation: str                # currently template string; LLM layer will overwrite this
    confidence: float
    color_tag: str                  # "yellow" | "red"

    # --- media / cross-references ---
    thumbnail_path: Optional[str] = None
    bbox_overlay: list[BBoxOverlayEntry] = Field(default_factory=list)
    heatmap_ref: Optional[str] = None
    related_event_id: Optional[str] = None


class Person(BaseModel):
    person_id: str
    video_id: str
    seat_id: Optional[str] = None
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    embedding_vector: Optional[List[float]] = None
    thumbnail_path: Optional[str] = None
    total_duration: Optional[float] = None


class CompleteSignal(BaseModel):
    video_id: str
    status: str = "done"
    total_events: Optional[int] = None