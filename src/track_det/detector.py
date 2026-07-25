"""
P2 module — Detection.
Owns the shared function contract:
    detect_objects(frame) -> [boxes, class, conf]
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

_detector = None
if _HAS_ULTRALYTICS and DEFAULT_WEIGHTS.exists():
    _detector = YOLO(str(DEFAULT_WEIGHTS))
    print(f"[detector] Loaded detector weights: {DEFAULT_WEIGHTS}")
else:
    print(f"[detector] Weights not found at {DEFAULT_WEIGHTS} "
          f"— detect_objects() will return an empty list until this is set.")


def detect_objects(frame) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
    """
    Contract: detect_objects(frame) -> [boxes, class, conf]
    """
    if _detector is None:
        return []

    results = _detector.predict(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = _detector.names[cls_id]
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2, y2), cls_name, conf))
    return detections
