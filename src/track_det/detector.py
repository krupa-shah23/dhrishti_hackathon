"""
P2 module — Detection.

Frozen contracts (final-implementation-plan-5day.md §2, P2 doc §1):

    detect_objects(roi_crops_over_window, exam_mode) -> [{"class": str, "confidence": float}]
        Input: a list of ROI crops from consecutive frames within ONE flagged
        window (P1 hands you a window, never single frames).
        Output: best-confidence-per-class results across the whole window,
        sorted highest confidence first. Empty list [] if nothing detected.
        P3 (Event dict owner) reads this as:
            result = detect_objects(crops, exam_mode)
            event["object_detected"]   = result[0]["class"] if result else None
            event["object_confidence"] = result[0]["confidence"] if result else 0.0

    detect_small_object_tiled(crop, exam_mode=None) -> [{"class": str, "confidence": float}]
        Sub-routine called BY detect_objects when a crop's best result is
        low-confidence or the box is tiny — NOT a separate pipeline stage.

RULE THAT MUST NEVER BE VIOLATED (P2 doc §4.6, §1):
    Your output never gates a flag — it only boosts severity. Never add an
    `if confidence < threshold: suppress_event()` line anywhere in this
    module. If P1's motion+MOG2 fired, the event fires, full stop — a low
    or missing detection here is a valid, honest result, not an error.
"""
from typing import List, Dict, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "phone_detector_v3.pt"

_detector = None
if _HAS_ULTRALYTICS and DEFAULT_WEIGHTS.exists():
    _detector = YOLO(str(DEFAULT_WEIGHTS))
    print(f"[detector] Loaded detector weights: {DEFAULT_WEIGHTS}")
else:
    print(f"[detector] Weights not found at {DEFAULT_WEIGHTS} "
          f"— detect_objects() will return [] until this is set.")

# --- tiling heuristics (tune against your held-out small-object subset, P2 doc §6) ---
TINY_BOX_AREA_FRACTION = 0.05   # box area < 5% of crop area -> "tiny"
LOW_CONFIDENCE_THRESHOLD = 0.45  # below this -> worth retrying at higher res
TILE_UPSCALE_FACTOR = 3

# --- per-class minimum detection confidence (confidence_curve.csv F1 sweep, "all" split) ---
# phone: 0.30 is the F1-maximizing threshold (0.9125) and edges out the near-tied
#   0.25 (0.912) on precision (0.9061 vs 0.8874) per class-1 confidence.
# paper-chit: 0.20, not the raw F1-max of 0.10 (0.9869) — chosen for precision
#   (0.9907 vs 0.9741) since object confidence here only affects severity, never
#   gates events, so the conservative side is lower-risk. Also: nearly all chit
#   ground truth (111/113 "all"-split instances) is in the train split the model
#   was fit on (valid has 2, test has 0), so the very-low-threshold end of that
#   curve is not backed by held-out data and shouldn't be trusted at face value.
# Edit this dict directly to retune; DEFAULT_MIN_CONFIDENCE covers any class
# not listed here (e.g. "calculator" once it's added) so a new class doesn't
# silently detect at conf=0.01.
MIN_DETECTION_CONFIDENCE = {
    "phone": 0.30,
    "paper-chit": 0.20,
}
DEFAULT_MIN_CONFIDENCE = 0.25


def _passes_class_threshold(cls_name: str, confidence: float) -> bool:
    return confidence >= MIN_DETECTION_CONFIDENCE.get(cls_name, DEFAULT_MIN_CONFIDENCE)


def class_filter(exam_mode: str) -> List[str]:
    """
    P2 doc §4.3 — exam-mode class gating. One if statement, lives here,
    called from detect_objects. Do not build a bigger config system than
    this needs.
    """
    if exam_mode == "CBT":
        return ["phone", "paper-chit"]
    elif exam_mode == "paper_pen":
        return ["phone"]           # paper-chit disabled — paper is legitimate everywhere
    elif exam_mode == "physical_fitness":
        return []                  # object detection off entirely — different problem
    else:
        # unknown/unset exam_mode: fail open to the full CBT class set rather
        # than silently detecting nothing — a missing mode shouldn't look
        # like "physical_fitness" (detection off) by accident.
        return ["phone", "paper-chit"]


def yolo_infer(crop, allowed_classes: List[str]) -> List[Dict]:
    """Runs YOLO on a single crop, filtered to allowed_classes. Returns
    a list of {"class", "confidence"} dicts, one per detected box that
    passes the class filter — NOT reduced/sorted yet, that happens in
    detect_objects()."""
    if _detector is None or not allowed_classes:
        return []

    # conf=0.01 here is just a floor to get raw candidate boxes out of YOLO —
    # the real filtering happens below via MIN_DETECTION_CONFIDENCE, which is
    # per-class and can't be expressed as a single conf= kwarg to predict().
    results = _detector.predict(crop, conf=0.01, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = _detector.names[cls_id]
            if cls_name not in allowed_classes:
                continue
            conf = float(box.conf[0])
            if not _passes_class_threshold(cls_name, conf):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name, "confidence": conf,
                "_box": (x1, y1, x2, y2),  # internal, used for tiny-box check
            })
    return detections


def is_low_confidence_or_tiny(crop, detections: List[Dict]) -> bool:
    """True if this crop's best result is weak enough to be worth a
    tiled re-infer (P2 doc §4.2 — chit-in-pen-cap case)."""
    if not detections:
        return True  # nothing found at all — worth a closer look before giving up

    best = max(detections, key=lambda d: d["confidence"])
    if best["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return True

    crop_area = crop.shape[0] * crop.shape[1] if hasattr(crop, "shape") else None
    if crop_area:
        x1, y1, x2, y2 = best["_box"]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area / crop_area < TINY_BOX_AREA_FRACTION:
            return True

    return False


def detect_small_object_tiled(crop, exam_mode: Optional[str] = None) -> List[Dict]:
    """
    P2 doc §4.2 — upsample the crop and re-infer at higher effective
    resolution. Same weights, no retraining. exam_mode is optional here
    (defaults to full class set) since the frozen top-level contract lists
    this as detect_small_object_tiled(crop) — callers invoking it directly
    (rather than via detect_objects) may not have exam_mode on hand.
    """
    import cv2

    allowed = class_filter(exam_mode) if exam_mode else ["phone", "paper-chit"]
    if _detector is None or not allowed:
        return []

    upsampled = cv2.resize(crop, None, fx=TILE_UPSCALE_FACTOR, fy=TILE_UPSCALE_FACTOR,
                            interpolation=cv2.INTER_CUBIC)
    return yolo_infer(upsampled, allowed)


def _max_confidence_per_class(all_detections: List[Dict]) -> List[Dict]:
    """Reduces a flat list of {"class","confidence",...} dicts (possibly
    with duplicate classes across multiple crops) down to one entry per
    class — the max confidence seen for that class anywhere in the window.
    Sorted highest confidence first. Strips the internal "_box" key."""
    best_per_class: Dict[str, float] = {}
    for d in all_detections:
        cls = d["class"]
        if d["confidence"] > best_per_class.get(cls, 0.0):
            best_per_class[cls] = d["confidence"]

    return [{"class": cls, "confidence": conf}
            for cls, conf in sorted(best_per_class.items(), key=lambda kv: kv[1], reverse=True)]


def detect_objects(roi_crops_over_window: List, exam_mode: str) -> List[Dict]:
    """
    FROZEN CONTRACT: detect_objects(roi_crops_over_window, exam_mode)
        -> [{"class": str, "confidence": float}]

    One call = one flagged window's object read. Runs every crop through
    YOLO (class-filtered by exam_mode), retries low-confidence/tiny results
    via detect_small_object_tiled(), then reduces everything to the best
    confidence seen per class across the whole window.

    Returns [] if nothing detected, or if exam_mode == "physical_fitness"
    (class_filter returns no classes -> nothing ever runs). An empty list
    is a valid, honest result — see the module docstring's rule on never
    gating a flag based on this output.
    """
    if _detector is None or not roi_crops_over_window:
        return []

    allowed = class_filter(exam_mode)
    if not allowed:
        return []

    all_detections: List[Dict] = []
    for crop in roi_crops_over_window:
        r = yolo_infer(crop, allowed)
        if is_low_confidence_or_tiny(crop, r):
            tiled = detect_small_object_tiled(crop, exam_mode)
            # keep whichever result is stronger, don't just overwrite —
            # tiling helps small objects but can occasionally be noisier
            # on already-clear ones, so take the better of the two per crop.
            r = tiled if _best_conf(tiled) > _best_conf(r) else r
        all_detections.extend(r)

    return _max_confidence_per_class(all_detections)


def _best_conf(detections: List[Dict]) -> float:
    return max((d["confidence"] for d in detections), default=0.0)


# --- kept for debugging/overlay tools that want raw single-frame boxes ---
def detect_objects_single_frame(frame, exam_mode: str = "CBT") -> List[Dict]:
    """Not part of the frozen contract. Convenience wrapper for scripts
    like overlay_frame_boxes.py that want one frame's raw detections
    rather than a window-reduced result."""
    return yolo_infer(frame, class_filter(exam_mode))