"""
severity_scoring.py
P1 — Severity Scoring (ML master doc §3.7)

Formula:
    S = w1*norm_motion + w2*norm_duration + w3*norm_repetition + w4*norm_obj_conf

Contract:
    - object_confidence is a *booster*, NEVER a gate: S > 0 even when
      object_confidence is 0 or None, provided motion + duration are non-zero.
    - All inputs are normalised to [0, 1] before weighting.
    - score_to_risk_label() maps the weighted sum → (risk_label, color_tag, confidence).

Integration contract (§9) fields produced:
    severity_score  : float in [0, 1]
    risk_label      : "low" | "medium" | "high"
    color_tag       : "yellow" | "red"
    confidence      : int 0-100
"""

from __future__ import annotations
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Default weights — chosen to match the spec's intent:
#   motion (w1=0.40) + duration (w2=0.35) dominate ≈ 75% of the score.
#   repetition (w3=0.15) adds evidence of sustained behaviour.
#   object_confidence (w4=0.10) boosts but never gates.
#
# Rationale:
#   · Motion intensity is the primary motion-gate signal that triggered the
#     event in the first place. High weight preserves that primacy.
#   · Duration encodes how long the suspicious behaviour lasted — directly
#     relevant to exam-hall context (brief movement ≠ cheating, 30s does).
#   · Repetition (number of detected-object hits) is a coarser proxy; useful
#     but noisier than raw motion, so lower weight.
#   · Object confidence is explicitly a booster per spec; 0.10 means a fully
#     confident detection (1.0) adds 0.10 to the score — meaningful bump, but
#     zero confidence contributes nothing and does not suppress the score.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "w1": 0.40,   # motion_intensity
    "w2": 0.35,   # duration
    "w3": 0.15,   # repetition_count
    "w4": 0.10,   # object_confidence
}

# Normalisation ceilings — anything above is clamped to 1.0.
# Chosen from real pipeline observations (clip 04 events):
#   motion_intensity: MOG2 fraction-of-pixels; > 0.50 is already very active.
#   duration:         events rarely exceed 60 s in a single track lifetime;
#                     even 30 s should score near 1.0.
#   repetition_count: > 10 detection hits in a track is unusually persistent.
#   object_confidence: already in [0,1] from YOLO (or None → 0).
NORM_CEIL = {
    "motion_intensity": 0.15,   # fraction of frame pixels; 15% is extremely active
    "duration":         30.0,   # seconds; 30 s sustained activity → max score
    "repetition_count": 10.0,   # detection hits; 10+ hits → max repetition score
    "object_confidence": 1.0,   # already 0–1
}


def _norm(value: float, ceil: float) -> float:
    """Clamp-then-scale: maps raw value → [0, 1] linearly, ceiling at `ceil`."""
    if ceil <= 0:
        return 0.0
    return min(1.0, max(0.0, value / ceil))


def compute_severity(
    motion_intensity: float,
    duration: float,
    repetition_count: float,
    object_confidence: Optional[float],
    weights: Optional[dict] = None,
) -> float:
    """
    Compute a weighted severity score in [0, 1].

    Parameters
    ----------
    motion_intensity : float
        Fraction of frame pixels flagged as foreground by motion.py's
        motion_intensity(mask). Raw domain: [0, 1] but typically < 0.15.
    duration : float
        Track lifetime in seconds.
    repetition_count : float
        Number of object-detection hits accumulated during the track's lifetime.
        Use 0 if no detections fired.
    object_confidence : float | None
        Max (or mean) YOLO confidence across all detections for this track.
        None or 0 → treated as 0 (booster absent, NOT a gate).
    weights : dict | None
        Override DEFAULT_WEIGHTS. Keys: w1, w2, w3, w4.

    Returns
    -------
    float : severity score in [0.0, 1.0]
    """
    w = DEFAULT_WEIGHTS if weights is None else weights

    # Guard: weights should sum to ≤ 1.0
    total_w = w["w1"] + w["w2"] + w["w3"] + w["w4"]
    if total_w <= 0:
        return 0.0

    # Normalise each input to [0, 1]
    n_motion = _norm(float(motion_intensity), NORM_CEIL["motion_intensity"])
    n_duration = _norm(float(duration), NORM_CEIL["duration"])
    n_repetition = _norm(float(repetition_count), NORM_CEIL["repetition_count"])

    # object_confidence: None or 0 → 0 (booster absent, not a gate)
    raw_conf = float(object_confidence) if object_confidence is not None else 0.0
    n_obj_conf = _norm(raw_conf, NORM_CEIL["object_confidence"])

    raw_score = (
        w["w1"] * n_motion
        + w["w2"] * n_duration
        + w["w3"] * n_repetition
        + w["w4"] * n_obj_conf
    )

    # Divide by the total weight so the score stays in [0, 1] even if custom
    # weights sum to something other than 1.0.
    score = raw_score / total_w
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Thresholds for risk classification.
# Calibrated so that:
#   · A short (< ~4 s), low-motion event scores < 0.25 → "low" / yellow
#   · A sustained (> ~10 s), active event scores ≥ 0.50 → "high" / red
#   · Anything in between → "medium" / yellow (flag for human review)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "high":   0.50,   # score ≥ 0.50 → high / red
    "medium": 0.25,   # score ≥ 0.25 → medium / yellow
    # else           → low / yellow
}


def score_to_risk_label(score: float) -> Tuple[str, str, int]:
    """
    Map a severity score → (risk_label, color_tag, confidence).

    Parameters
    ----------
    score : float in [0, 1]

    Returns
    -------
    risk_label  : "low" | "medium" | "high"
    color_tag   : "yellow" | "red"
    confidence  : int 0-100
        Linear mapping of the score within the event's label band:
        · low    band [0, 0.25)  → confidence in [0, 49]
        · medium band [0.25, 0.5) → confidence in [50, 74]
        · high   band [0.5, 1.0] → confidence in [75, 100]
    """
    score = min(1.0, max(0.0, float(score)))

    if score >= THRESHOLDS["high"]:
        risk_label = "high"
        color_tag = "red"
        # Map [0.5, 1.0] → [75, 100]
        band_frac = (score - 0.50) / 0.50
        confidence = int(75 + band_frac * 25)
    elif score >= THRESHOLDS["medium"]:
        risk_label = "medium"
        color_tag = "yellow"
        # Map [0.25, 0.5) → [50, 74]
        band_frac = (score - 0.25) / 0.25
        confidence = int(50 + band_frac * 24)
    else:
        risk_label = "low"
        color_tag = "yellow"
        # Map [0, 0.25) → [0, 49]
        band_frac = score / 0.25
        confidence = int(band_frac * 49)

    return risk_label, color_tag, confidence
