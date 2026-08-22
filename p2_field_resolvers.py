"""
Drop-in additions to close 3 of the TODO fields in event_adapter.py:
  camera_id, exam_phase, repetition_count

All three are computable entirely on P2's side — no dependency on P1's
reply about motion-intensity fields.

ONE DESIGN ASSUMPTION FLAGGED, NOT SILENTLY DECIDED (confirm before shipping
to slides/final report):
  repetition_count's definition below is "number of distinct detection
  bursts (non-contiguous runs of object_detected=True) within this single
  finalized event" — a proxy built from data P2P3Bridge already has
  (per-frame object-detection flags across the track's lifetime).
  This is NOT necessarily what the ML master doc's severity formula meant
  by "repetition" — P1's own doc describes a related but different concept
  ("3+ repeated identical angular snaps within a 2-min window" from her
  pose/gesture signal, which is cross-event/gesture-pattern repetition,
  not single-event burst count). Ship this as a stated proxy on slides if
  you don't have time to reconcile it against her definition, don't imply
  it's the same thing.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# camera_id: clip -> camera mapping
# ---------------------------------------------------------------------------
# Fill this in using your own already-resolved clip-identity mapping
# (the one built for ground_truth_events.csv — clip1..clip8 -> real filenames).
# Falls back to the existing f"cam_{video_id}" default if a video_id isn't
# in the map, so nothing breaks if a clip is missing.
CAMERA_ID_MAP: Dict[str, str] = {
    # "07_seat_exchange": "cam_A",
    # "06_phone_use": "cam_B",
    # ... fill in from your clip <-> filename <-> camera mapping
}


def resolve_camera_id(video_id: str) -> str:
    return CAMERA_ID_MAP.get(video_id, f"cam_{video_id}")


# ---------------------------------------------------------------------------
# exam_phase: purely time-position-based, no P1 dependency
# ---------------------------------------------------------------------------
# Matches the ML master doc's phase boundaries used in P1's motion
# multipliers (Phase 1: 0-10% distribution, Phase 2: 10-90% core,
# Phase 3: 90-100% submission) — computed independently here from the
# event's own start time and the video's total duration, which
# P2P3Bridge/event_adapter already has access to (or can get via the
# same video-metadata lookup used elsewhere in the pipeline).
def resolve_exam_phase(event_start: float, video_duration: float) -> str:
    if video_duration <= 0:
        return "main_exam"  # can't compute position, fall back to existing default
    position = event_start / video_duration
    if position <= 0.10:
        return "distribution"
    if position >= 0.90:
        return "submission"
    return "core"


# ---------------------------------------------------------------------------
# repetition_count: burst-count proxy (see flagged assumption above)
# ---------------------------------------------------------------------------
def compute_repetition_count(per_frame_object_detected: List[Optional[bool]]) -> int:
    """
    Counts distinct contiguous runs of True in a per-frame object_detected
    sequence for one finalized event/track. E.g. [F,T,T,F,F,T,F,T,T,T] -> 3 bursts.

    Requires P2P3Bridge to expose the per-frame object_detected list for
    the track before finalization collapses it to a single bool — check
    whether that per-frame list is already retained anywhere (e.g. as an
    internal list on the track dict before _finalize_track() collapses it)
    or whether this needs a small addition to P2P3Bridge to retain it.
    """
    if not per_frame_object_detected:
        return 0
    count = 0
    prev = False
    for flag in per_frame_object_detected:
        cur = bool(flag)
        if cur and not prev:
            count += 1
        prev = cur
    return count