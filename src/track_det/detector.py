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

# Path to your fine-tuned weights (Day 1 output).
# Resolved relative to THIS FILE's location, not the current working
# directory — this file lives at <repo_root>/src/track_det/detector.py,
# so we go up 3 levels to reach <repo_root>, then into models/.
# (A plain Path("models/phone_detector_v1.pt") broke depending on
# whether you launched from repo root or from inside src/track_det/ —
# this version works regardless of cwd.)
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

    frame: a single video frame (numpy array, BGR — as read by cv2).
    returns: list of (box, class_name, confidence) tuples, e.g.:
             [((x1, y1, x2, y2), "phone", 0.93), ...]
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