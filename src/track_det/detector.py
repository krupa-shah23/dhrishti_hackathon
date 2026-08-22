"""
P2 module — Detection.
Owns the shared function contract:
    detect_objects(roi_crops_over_window, exam_mode) -> [boxes, class, conf]
"""
from typing import List, Tuple
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
    print(f"[detector] Loaded detector weights: {DEFAULT_WEIGHTS}")
else:
    print(f"[detector] Weights not found at {DEFAULT_WEIGHTS} "
          f"— detect_objects() will return an empty list until this is set.")


def detect_objects(roi_crops_over_window: List, exam_mode: str) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
    """
    Contract: detect_objects(roi_crops_over_window, exam_mode) -> [boxes, class, conf]

    Returns detections with conf >= MIN_DETECTION_CONFIDENCE only.
    See constant definition above for threshold rationale and accepted tradeoffs.
    """
    if _detector is None or not roi_crops_over_window:
        return []

    # YOLO predict accepts a list of images natively
    results = _detector.predict(roi_crops_over_window, verbose=False)
    detections = []

    # We aggregate detections across the entire window
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < MIN_DETECTION_CONFIDENCE:
                continue  # filtered: below P2's confidence threshold
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = _detector.names[cls_id]
            detections.append(((x1, y1, x2, y2), cls_name, conf))

    # Remove duplicates or overlapping boxes from multiple frames if needed,
    # but for now we return all detections found in the window.
    return detections
