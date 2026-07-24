# Frozen Function Contracts (Day 1)

These function contracts are frozen across all 4 module teams (P1, P2, P3, P4) to guarantee seamless pipeline integration in `main.py`.

---

## 1. P1 — Motion & ROI (`src/motion`)

```python
def get_motion_mask(frame: np.ndarray) -> np.ndarray:
    """
    Computes a binary foreground motion mask from a single video frame.

    Args:
        frame (np.ndarray): BGR image frame (H, W, 3).

    Returns:
        mask (np.ndarray): Binary motion mask (H, W) where 255 represents motion.
    """
    pass

def get_rois(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Extracts bounding boxes (ROIs) around motion regions from a binary mask.

    Args:
        mask (np.ndarray): Binary motion mask (H, W).

    Returns:
        boxes (list[tuple]): List of ROI bounding boxes in (x, y, w, h) format.
    """
    pass
```

---

## 2. P2 — Tracking & Detection (`src/track_det`)

```python
def track(boxes: list[tuple[int, int, int, int]]) -> list[dict]:
    """
    Tracks bounding boxes across consecutive frames using ByteTrack or Centroid fallback.

    Args:
        boxes (list[tuple]): List of detected ROI bounding boxes (x, y, w, h).

    Returns:
        tracks (list[dict]): List of track dicts with keys ['track_id', 'box', 'history'].
    """
    pass

def detect_objects(frame: np.ndarray) -> list[tuple[list[int], str, float]]:
    """
    Detects prohibited objects (mobile phone, paper/chits) using fine-tuned YOLO.

    Args:
        frame (np.ndarray): BGR image frame.

    Returns:
        detections (list[tuple]): List of ([x1, y1, x2, y2], class_name, confidence).
    """
    pass
```

---

## 3. P3 — Features & Risk Scoring (`src/features_risk`)

```python
def extract_features(track: dict) -> dict:
    """
    Extracts spatial-temporal movement features from a tracked entity.

    Args:
        track (dict): Track dictionary containing box history and movement vectors.

    Returns:
        features (dict): Dict of extracted features (velocity, variance, aspect_ratio_change, etc.).
    """
    pass

def risk_score(features: dict) -> float:
    """
    Computes a behavioral risk score between 0.0 and 1.0 using XGBoost risk model.

    Args:
        features (dict): Extracted feature dictionary.

    Returns:
        score (float): Risk score in range [0.0, 1.0].
    """
    pass
```

---

## 4. P4 — CSV Event Log Schema

Output location: `outputs/<video_id>_events.csv`

| Field | Type | Description |
|---|---|---|
| `timestamp` | `str` | Video timestamp (HH:MM:SS.mmm) |
| `frame_index` | `int` | Current frame number |
| `roi_box` | `str` | Formatted string `"(x,y,w,h)"` |
| `motion_score` | `float` | Motion intensity score (0.0 to 100.0) |
| `audio_level` | `float` | Audio decibel / energy level |
| `risk_score` | `float` | Behavioral risk score (0.0 to 1.0) |
| `confidence` | `float` | Overall event confidence score |
