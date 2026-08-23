"""
P2 module (joint with P1, per Day 4 plan) — Invigilator-motion exclusion.

Not one of the original 6 shared function contracts — this is a new
filter that sits between track() output and P3's extract_features(),
same relationship fuse_track_detections() has to the pipeline.

Core idea (from the team's brainstorm doc): the single largest source
of motion in real exam-hall footage is the invigilator patrolling the
aisles. A seated student's motion stays confined to a small area
(fidgeting, head turns) — an invigilator's track covers a lot of
ground. We don't need a new detector for this: CentroidTracker already
gives us track.history (list of boxes per frame) for free, so this is
pure post-hoc analysis of data we already have.

THIS IS A PROTOTYPE, NOT TUNED: thresholds below (path_ratio, min_hits)
are guesses based on the synthetic test in this file, not real
footage. Do not treat the specific numbers as final — re-tune once
real multi-person exam-hall clips are available from P1.
"""

from typing import List, Tuple
import math


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(history: List[Tuple[float, float, float, float]]) -> float:
    """
    Total centroid-to-centroid distance traveled across a track's
    history — sum of per-frame displacement, NOT straight-line distance
    from start to end. A seated student fidgeting back and forth covers
    real path length even if their NET displacement is ~0, so summing
    per-step distance (rather than start-to-end distance) is what
    actually distinguishes "moves around a lot in a small area" from
    "moves around a lot across a wide area" — see net_displacement()
    below for the complementary check.
    """
    if len(history) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(history)):
        total += _dist(_centroid(history[i - 1]), _centroid(history[i]))
    return total


def net_displacement(history: List[Tuple[float, float, float, float]]) -> float:
    """
    Straight-line distance between the FIRST and LAST centroid in the
    history. A patrolling invigilator has high net displacement (ends
    up far from where they started); a seated, fidgeting student has
    low net displacement (ends up back where they started) even if
    their path_length() is nontrivial.
    """
    if len(history) < 2:
        return 0.0
    return _dist(_centroid(history[0]), _centroid(history[-1]))


def is_invigilator_track(
    history: List[Tuple[float, float, float, float]],
    min_hits: int = 15,
    min_net_displacement: float = 150.0,
    min_path_length: float = 150.0,
) -> bool:
    """
    Flags a track as invigilator-like (vs seated-student-like) based on
    movement pattern alone — no appearance/detection model needed.

    A track is flagged as invigilator ONLY if BOTH conditions hold:
      1. net_displacement >= min_net_displacement
         (ended up meaningfully far from where it started — rules out
         a student who fidgets a lot but stays in their seat)
      2. path_length >= min_path_length
         (covered real ground, not just one big jump/track-ID error)

    Requiring BOTH, not just one, matters: a track with high path_length
    but low net_displacement is a fidgeter (moves a lot, stays put) —
    exactly the seated-student case we must NOT flag. A track with high
    net_displacement but very short history could just be a tracking
    glitch/ID jump, not a real patrol — min_hits guards against that.

    min_hits: minimum track history length before this heuristic is
              even evaluated. Too short a history isn't enough evidence
              either way (a 2-frame track jump looks identical to real
              motion) — return False (not-invigilator) rather than
              guess on insufficient data.

    THESE DEFAULTS ARE PLACEHOLDERS. They were picked to make the
    synthetic test below discriminate cleanly, not calibrated against
    real footage. Expect to change min_net_displacement/min_path_length
    once you know the real camera's resolution and the aisle's actual
    pixel length in frame.
    """
    if len(history) < min_hits:
        return False

    disp = net_displacement(history)
    path = path_length(history)

    return disp >= min_net_displacement and path >= min_path_length