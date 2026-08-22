"""
P2 module — Detection <-> Track fusion.

`track(boxes)` and `detect_objects(frame)` are two independent per-frame
outputs — tracking runs on P1's motion ROI boxes, detection runs YOLO on
the raw frame. Neither knows about the other. This module is the missing
link: for a given frame, attach the best-matching detection (class,
confidence) to each track, so downstream (P3's extract_features) can ask
"was a phone detected on track X at time T" instead of getting two
disconnected lists.

UPDATED: matching changed from per-detection greedy (each detection picks
its own best-containment track, resolved by highest confidence if two
detections compete for the same track) to Hungarian assignment
(scipy.optimize.linear_sum_assignment) -- same technique used to fix
centroid_tracker.py's matching. This finds the globally optimal one-to-one
track<->detection pairing by containment cost, rather than resolving
competing claims greedily/sequentially.

Not part of the original 6 shared function contracts -- this is a P2-side
helper called once per frame, after both track() and detect_objects()
have been called for that frame.
"""
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .invigilator_filter import is_invigilator_track


def _intersection_area(box_a: Tuple[float, float, float, float],
                        box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _containment(det_box: Tuple[float, float, float, float],
                  track_box: Tuple[float, float, float, float]) -> float:
    """
    Fraction of the DETECTION box's area that falls inside the track box.
    Deliberately not plain IoU: track boxes come from motion ROIs and are
    typically much larger/looser than a tight YOLO phone box, so a phone
    box fully sitting inside a person-sized motion blob scores a LOW IoU
    (union is dominated by the track's size) even though it's a perfect
    containment match. Containment-of-the-smaller-box is the right test
    for "is this detection happening within this track's region", which
    is what we actually want to know.
    """
    dx1, dy1, dx2, dy2 = det_box
    det_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
    if det_area == 0.0:
        return 0.0
    return _intersection_area(det_box, track_box) / det_area


def _resolve_invigilator_flag(
    t: Dict[str, Any],
    track_histories: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    invigilator_flags: Optional[Dict[int, bool]] = None,
    min_hits: int = 15,
    min_net_displacement: float = 150.0,
    min_path_length: float = 150.0,
) -> bool:
    """
    Determines invigilator status for a track dictionary following strict precedence:
      1. Explicit invigilator_flags dict: if provided and track_id is present, wins.
      2. Computed from track histories: if track_histories dict (e.g. from
         tracker.get_track_history(tid)) or t['history'] is available, evaluated
         via is_invigilator_track().
      3. Pre-existing 'invigilator_flag' key on the track dict.
      4. Default fallback: False.
    """
    tid = t.get("track_id")

    # Precedence 1: Explicit invigilator_flags dict
    if invigilator_flags is not None and tid in invigilator_flags:
        return bool(invigilator_flags[tid])

    # Precedence 2: Computed from track histories
    if track_histories is not None and tid in track_histories:
        return is_invigilator_track(
            track_histories[tid],
            min_hits=min_hits,
            min_net_displacement=min_net_displacement,
            min_path_length=min_path_length,
        )
    if "history" in t and t["history"] is not None:
        return is_invigilator_track(
            t["history"],
            min_hits=min_hits,
            min_net_displacement=min_net_displacement,
            min_path_length=min_path_length,
        )

    # Precedence 3: Pre-existing 'invigilator_flag' on track dict
    if "invigilator_flag" in t and t["invigilator_flag"] is not None:
        return bool(t["invigilator_flag"])

    # Precedence 4: Default fallback
    return False


def merge_invigilator_flags(
    fused_tracks: List[Dict[str, Any]],
    track_histories: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    invigilator_flags: Optional[Dict[int, bool]] = None,
    min_hits: int = 15,
    min_net_displacement: float = 150.0,
    min_path_length: float = 150.0,
) -> List[Dict[str, Any]]:
    """
    Attaches/updates 'invigilator_flag': bool for each fused track dictionary in-place.
    """
    for ft in fused_tracks:
        ft["invigilator_flag"] = _resolve_invigilator_flag(
            ft,
            track_histories=track_histories,
            invigilator_flags=invigilator_flags,
            min_hits=min_hits,
            min_net_displacement=min_net_displacement,
            min_path_length=min_path_length,
        )
    return fused_tracks


def fuse_track_detections(
    tracks: List[Dict[str, Any]],
    detections: List[Tuple[Tuple[float, float, float, float], str, float]],
    containment_thresh: float = 0.5,
    # NOTE: p1_p2_tracker.py currently sets invigilator_flag manually after calling
    # fuse_track_detections() (precedence tier 3 / post-fusion update). In the future,
    # P1 may simplify her call site by passing track_histories directly here.
    track_histories: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    invigilator_flags: Optional[Dict[int, bool]] = None,
    min_hits: int = 15,
    min_net_displacement: float = 150.0,
    min_path_length: float = 150.0,
) -> List[Dict[str, Any]]:
    """
    tracks:      output of track(boxes) for THIS frame
                 [{"track_id": int, "box": (x1,y1,x2,y2)}, ...]
    detections:  output of detect_objects(frame) for THIS frame
                 [((x1,y1,x2,y2), class_name, conf), ...]
    containment_thresh: minimum fraction of the detection box that must
                 fall inside a track's box to count as belonging to it.
    track_histories: optional mapping {track_id: [(x1,y1,x2,y2), ...]} from
                 tracker.get_track_history(tid).
    invigilator_flags: optional precomputed mapping {track_id: bool}.

    Returns tracks with extra keys merged in: "class", "confidence",
    and "invigilator_flag" (bool). Uses Hungarian assignment to
    find the globally optimal one-to-one track<->detection pairing by
    containment cost, so a track is never assigned a worse-matching
    detection just because a better-matching detection was processed
    later, and a detection is never "stolen" from its best track by a
    tie-break ordering artifact.
    """
    fused = [
        {
            "track_id": t["track_id"],
            "box": t["box"],
            "class": None,
            "confidence": None,
            "invigilator_flag": _resolve_invigilator_flag(
                t,
                track_histories=track_histories,
                invigilator_flags=invigilator_flags,
                min_hits=min_hits,
                min_net_displacement=min_net_displacement,
                min_path_length=min_path_length,
            ),
        }
        for t in tracks
    ]

    if not tracks or not detections:
        return fused

    detections = sorted(detections, key=lambda d: d[2], reverse=True)

    n_tracks = len(tracks)
    n_dets = len(detections)

    # Cost = 1 - containment (Hungarian minimizes cost, we want to
    # maximize containment). Pairs below threshold get an infeasible
    # cost so linear_sum_assignment naturally avoids them.
    INFEASIBLE_COST = 1e9
    cost_matrix = np.full((n_tracks, n_dets), INFEASIBLE_COST)

    for i, t in enumerate(tracks):
        for j, (det_box, cls_name, conf) in enumerate(detections):
            containment = _containment(det_box, t["box"])
            if containment >= containment_thresh:
                cost_matrix[i, j] = 1.0 - containment

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < INFEASIBLE_COST:
            det_box, cls_name, conf = detections[col]
            fused[row]["class"] = cls_name
            fused[row]["confidence"] = conf

    return fused