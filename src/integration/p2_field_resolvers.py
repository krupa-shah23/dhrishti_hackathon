"""
p2_field_resolvers.py
Field resolvers for event_adapter: camera_id, exam_phase, repetition_count.

Computable entirely on P2's side without external P1 dependency.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# camera_id: clip -> camera mapping
# ---------------------------------------------------------------------------
# Falls back to the existing f"cam_{video_id}" default if a video_id isn't
# in the map, so nothing breaks if a clip is missing.
CAMERA_ID_MAP: Dict[str, str] = {
    "clip7": "cam_clip7",
    "07_seat_exchange": "cam_clip7",
}


def resolve_camera_id(video_id: str) -> str:
    """
    Resolves video_id/clip_name to camera_id. Falls back to f"cam_{video_id}".
    """
    if not video_id:
        return "cam_unknown"
    return CAMERA_ID_MAP.get(video_id, f"cam_{video_id}")


# ---------------------------------------------------------------------------
# exam_phase: purely time-position-based
# ---------------------------------------------------------------------------
# Matches the ML master doc's phase boundaries:
#   Phase 1: 0-10% distribution
#   Phase 2: 10-90% core
#   Phase 3: 90-100% submission
def resolve_exam_phase(event_start: float, video_duration: float) -> str:
    """
    Determines exam_phase based on event start timestamp and total video duration.
    Falls back to 'main_exam' if video_duration is unknown (<= 0).
    """
    if video_duration <= 0:
        return "main_exam"  # can't compute position, fall back to existing default
    position = event_start / video_duration
    if position <= 0.10:
        return "distribution"
    if position >= 0.90:
        return "submission"
    return "core"


# ---------------------------------------------------------------------------
# repetition_count: burst-count proxy
# ---------------------------------------------------------------------------
# NOT CURRENTLY REACHABLE — P2P3Bridge doesn't retain per-frame object_detected
# history. Decision (2026-08-22): ship repetition_count as disclosed 0, not
# building retention. See PROJECT_SCOPE_DECISIONS.md.
def compute_repetition_count(per_frame_object_detected: List[Optional[bool]]) -> int:
    """
    Counts distinct contiguous runs of True in a per-frame object_detected
    sequence for one finalized event/track. E.g. [F,T,T,F,F,T,F,T,T,T] -> 3 bursts.
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
