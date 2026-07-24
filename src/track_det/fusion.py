"""
P2 module — Detection <-> Track fusion.

`track(boxes)` and `detect_objects(frame)` are two independent per-frame
outputs — tracking runs on P1's motion ROI boxes, detection runs YOLO on
the raw frame. Neither knows about the other. This module is the missing
link: for a given frame, attach the best-matching detection (class,
confidence) to each track, so downstream (P3's extract_features) can ask
"was a phone detected on track X at time T" instead of getting two
disconnected lists.

Not part of the original 6 shared function contracts — this is a P2-side
helper called once per frame, after both track() and detect_objects()
have been called for that frame.
"""
from typing import List, Tuple, Dict, Any, Optional


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


def fuse_track_detections(
    tracks: List[Dict[str, Any]],
    detections: List[Tuple[Tuple[float, float, float, float], str, float]],
    containment_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    tracks:      output of track(boxes) for THIS frame
                 [{"track_id": int, "box": (x1,y1,x2,y2)}, ...]
    detections:  output of detect_objects(frame) for THIS frame
                 [((x1,y1,x2,y2), class_name, conf), ...]
    containment_thresh: minimum fraction of the detection box that must
                 fall inside a track's box to count as belonging to it.
                 0.5 = at least half the phone box is within the track's
                 motion region.

    Returns tracks with two extra keys merged in:
        "class": str or None
        "confidence": float or None
    If multiple detections overlap the same track above threshold, keeps
    the highest-confidence one. If one detection overlaps multiple tracks
    (e.g. a phone box spanning two adjacent people), it's assigned to
    the track with the highest IoU only — not double-counted.
    """
    fused = [
        {"track_id": t["track_id"], "box": t["box"], "class": None, "confidence": None}
        for t in tracks
    ]

    # best (iou, det_conf) seen so far per track index, so a later
    # lower-confidence detection can't overwrite an earlier better match
    best_score: Dict[int, Tuple[float, float]] = {}

    for det_box, cls_name, conf in detections:
        best_track_idx: Optional[int] = None
        best_containment = 0.0
        for i, t in enumerate(fused):
            containment = _containment(det_box, t["box"])
            if containment >= containment_thresh and containment > best_containment:
                best_containment = containment
                best_track_idx = i

        if best_track_idx is None:
            continue  # detection doesn't fall inside any current track well enough

        prev = best_score.get(best_track_idx)
        if prev is None or conf > prev[1]:
            best_score[best_track_idx] = (best_containment, conf)
            fused[best_track_idx]["class"] = cls_name
            fused[best_track_idx]["confidence"] = conf

    return fused