"""
P2 module — Detection.
Frozen contract:

    detect_objects(roi_crops_over_window, exam_mode)
        -> [((x1, y1, x2, y2), class_name, confidence), ...]
        Input: a list of ROI crops from consecutive frames within ONE flagged
        window (P1 hands you a window, never single frames). Boxes are in the
        coordinate space of the crop they were found in; the caller translates
        them back to absolute frame coordinates.
        Output: best-confidence-per-class across the whole window, sorted
        highest confidence first. Empty list [] if nothing detected.

    detect_small_object_tiled(crop, exam_mode=None) -> [{"class","confidence","_box"}]
        Sub-routine for the low-confidence / tiny-box retry path — NOT a
        separate pipeline stage. Internal dict shape.

RULE THAT MUST NEVER BE VIOLATED (P2 doc §4.6, §1):
    Your output never gates a flag — it only boosts severity. Never add an
    `if confidence < threshold: suppress_event()` line anywhere in this
    module. If P1's motion+MOG2 fired, the event fires, full stop — a low
    or missing detection here is a valid, honest result, not an error.
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "phone_detector_v1.pt"

# Minimum confidence threshold for detections returned from detect_objects().
#
# Source: P2's F1-confidence curve analysis on the phone_detector_v1 model:
#   - phone-class F1 plateau: 0.20 – 0.45 (flat, not discriminative in this range)
#   - combined-class F1 optimum: 0.276
#   - known FP band: 0.27 – 0.47  (noise detections and spurious boxes concentrate here)
#   - known real-detection band: 0.41 – 0.52 (genuine phone/chit detections)
#
# 0.35 was chosen by P2 as a practical operating point: it cuts the dense FP band
# (0.27-0.35) while preserving detections in the real-detection band (0.41-0.52).
#
# Accepted tradeoff: the residual FP band 0.35–0.47 still passes through — detections
# in that range may be real or may be noise, and cannot be separated by threshold alone.
# Since object confidence never gates the Stage-A motion flag, worst-case impact of any
# remaining noise is inflated severity scoring downstream (P3's concern), NOT a false
# event being created from nothing or a real event being dropped.
#
# Do not lower this below 0.27 without re-validating the F1-confidence curve.
MIN_DETECTION_CONFIDENCE = 0.35

_detector = None
if _HAS_ULTRALYTICS and DEFAULT_WEIGHTS.exists():
    _detector = YOLO(str(DEFAULT_WEIGHTS))
    # CPU forced. GPU was re-validated with numpy<2 pinned (the earlier
    # segfault's real cause -- see below) and is genuinely faster when it
    # works (0.14-1.19s/frame vs CPU's growing-but-under-2s per frame in
    # the same run), but it STILL crashes: 2/2 GPU re-test runs segfaulted
    # inside torch.nn.functional.silu (YOLO's SiLU activation forward,
    # same signature both times) -- once during import before frame 0,
    # once at frame 12 after 11 real successful frames. numpy<2 delayed
    # the CPU crash (frame 3 -> ~14) and delayed/varied the GPU crash too,
    # but did not eliminate it on GPU. This points to a real, additional
    # GPU-specific instability (likely CUDA memory/context handling across
    # repeated inference calls on this RTX 3050 6GB + mediapipe's CPU
    # delegate sharing the process) beyond the numpy ABI issue -- not
    # something to re-attempt without deeper native-level debugging
    # (pinned CUDA/cuDNN/torch versions, or isolating YOLO into a separate
    # process from mediapipe). Do not re-flip this without that.
    _DETECTOR_DEVICE = "cpu"
    _detector.to(_DETECTOR_DEVICE)
    print(f"[detector] Loaded detector weights: {DEFAULT_WEIGHTS} (device={_DETECTOR_DEVICE})")
else:
    print(f"[detector] Weights not found at {DEFAULT_WEIGHTS} "
          f"— detect_objects() will return [] until this is set.")

# --- tiling heuristics (tune against your held-out small-object subset, P2 doc §6) ---
TINY_BOX_AREA_FRACTION = 0.05   # box area < 5% of crop area -> "tiny"
LOW_CONFIDENCE_THRESHOLD = 0.45  # below this -> worth retrying at higher res
TILE_UPSCALE_FACTOR = 3


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

    results = _detector.predict(crop, verbose=False)
    detections = []

    # We aggregate detections across the entire window
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = _detector.names[cls_id]
            if cls_name not in allowed_classes:
                continue
            conf = float(box.conf[0])
            if conf < MIN_DETECTION_CONFIDENCE:
                continue  # below P2's confidence threshold
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


def _max_confidence_per_class(all_detections: List[Dict]) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
    """Reduces a flat list of {"class","confidence","_box"} dicts (possibly
    with duplicate classes across multiple crops) down to one entry per
    class — the max confidence seen for that class anywhere in the window,
    keeping that detection's box. Returns the frozen tuple shape
    ((x1,y1,x2,y2), class_name, confidence), highest confidence first."""
    best: Dict[str, Dict] = {}
    for d in all_detections:
        cls = d["class"]
        if d["confidence"] > best.get(cls, {}).get("confidence", 0.0):
            best[cls] = d
    return [(d.get("_box", (0.0, 0.0, 0.0, 0.0)), cls, d["confidence"])
            for cls, d in sorted(best.items(), key=lambda kv: kv[1]["confidence"], reverse=True)]


def detect_objects(roi_crops_over_window: List, exam_mode: str) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
    """
    FROZEN CONTRACT: detect_objects(roi_crops_over_window, exam_mode)
        -> [((x1, y1, x2, y2), class_name, confidence), ...]

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