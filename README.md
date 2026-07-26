# Tracking & Detection (P2)

This module implements the **Tracking & Detection** pipeline for the Dhristi Offline Exam Hall Video Analytics system.

It is responsible for:

- Multi-object student tracking
- Phone detection using YOLOv8
- Associating detections with tracked students
- Invigilator filtering
- Tracking evaluation and stress testing

---

## Module Overview

```
Motion ROIs (P1)
        │
        ▼
   Student Tracking
        │
        ▼
 Phone Detection (YOLOv8)
        │
        ▼
Track–Detection Fusion
        │
        ▼
Track Features → Risk Model (P3)
```

---

## Features

### Student Tracking

- Centroid-based multi-object tracker
- Hungarian assignment for optimal matching
- Automatic track creation and deletion
- Configurable `max_distance`
- Configurable `max_age`
- Tracker reset functionality

---

### Phone Detection

YOLOv8n fine-tuned for phone detection.

Outputs:

```python
[(bounding_box, class_name, confidence), ...]
```

Performance on validation dataset:

| Metric | Score |
|---------|------:|
| Precision | 0.942 |
| Recall | 0.961 |
| mAP50 | 0.956 |
| mAP50-95 | 0.872 |

---

### Track–Detection Fusion

Associates detected objects with tracked students.

Instead of IoU, the module uses **containment-based matching**, making it robust when object boxes are significantly smaller than person tracking boxes.

Outputs:

- Track ID
- Detected object
- Confidence

---

### Invigilator Filter

Prototype module to distinguish invigilators from seated students using movement statistics.

Current features:

- Path length
- Net displacement
- Minimum track duration

---

## Folder Structure

```
track_det/
│
├── tracker.py
├── centroid_tracker.py
├── detector.py
├── fusion.py
├── invigilator_filter.py
│
├── test_track.py
├── test_track_stress.py
├── test_detect_webcam.py
├── test_fusion_sequence.py
├── test_invigilator_filter.py
│
└── run_on_p1_rois.py
```

---

## Installation

Create a virtual environment

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

Install dependencies

```bash
uv sync
```

---

## Running

### Tracker Test

```bash
uv run python -m src.track_det.test_track
```

### Stress Test

```bash
uv run python -m src.track_det.test_track_stress
```

### Webcam Detection

```bash
uv run python -m src.track_det.test_detect_webcam
```

### Fusion Test

```bash
uv run python -m src.track_det.test_fusion_sequence
```

### Invigilator Filter Test

```bash
uv run python -m src.track_det.test_invigilator_filter
```

### Run on ROI CSV

```bash
uv run python -m src.track_det.run_on_p1_rois
```

---

## Module Outputs

### Tracker

```python
track(boxes)
```

Returns

```python
[
    {
        "id": int,
        "bbox": [x1, y1, x2, y2]
    }
]
```

---

### Detector

```python
detect_objects(frame)
```

Returns

```python
[
    (bbox, class_name, confidence)
]
```

---

### Fusion

Associates detections with active tracks.

Output example:

```python
Track 5
Phone
Confidence: 0.91
```

---

## Current Status

- Student tracker implemented
- Hungarian matching integrated
- YOLOv8 phone detector trained
- Detection–tracking fusion completed
- Invigilator filter prototype completed
- Synthetic stress tests completed
- Webcam inference tested
- ROI CSV integration completed

---

## Future Improvements

- Kalman Filter tracking
- Appearance-based ReID
- Multi-camera tracking
- Multi-class prohibited object detection
- Real exam hall benchmarking