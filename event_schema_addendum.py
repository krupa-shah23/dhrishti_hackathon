"""
Drop-in replacement/addition for the Event model in
src/integration/fastapi_bridge/schemas.py

Locked decisions preserved (do NOT change without checking downstream):
  - object_detected is bool, NOT str|None (P3's extract_features() tests
    depend on this — 7/7 passing test suite)
  - object_confidence is a severity input, never a gate — can be None
  - color_tag is "yellow" | "red" only (no other values)

OPEN QUESTIONS — flagged, not resolved here, confirm before merging:
  1. `timestamps: [{start, end}]` (list, per ML master doc §9A) vs. the
     currently-implemented flat `start: float / end: float` (per your own
     session log's thin-version field list). This file keeps flat start/end
     since that's what's actually wired downstream right now — but if P1's
     event-splitting work (ground-truth log §4.2) ever produces multi-interval
     events, you'll need to migrate to the list form. Don't silently pick
     one — confirm with whoever else reads events off this schema.
  2. `risk_label` (this file, matches ML doc's original severity-scoring
     field) vs. `risk_level` (the NEW field name used in your own LLM
     reasoning layer doc, §11.3, with values low/medium/high). These may be
     meant to be the same field under two names, or genuinely two separate
     fields (rule-based risk_label as a kept fallback + LLM risk_level as
     the new primary). This file adds BOTH, with risk_label kept as the
     rule-based fallback per your own instruction ("keep the existing
     rule-based severity_score as a fallback/sanity-check field, not
     delete it") — confirm this is what you actually want before shipping.
  3. bbox_overlay's coordinate convention (pixel space vs. normalized) has
     never been confirmed against P1's tracking output — the field shape
     below matches the documented contract, but the coordinate space is
     still an open item per your own doc (§4.2 of the joint P1xP2 doc).
"""

from __future__ import annotations
from typing import Optional
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