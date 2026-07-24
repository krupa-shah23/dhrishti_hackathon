# Motion & ROI Detection Module

This module implements the **Motion & ROI (Region of Interest) Detection** stage of the Offline Exam-Hall Video Analytics pipeline. Its purpose is to detect meaningful motion in surveillance footage, suppress background noise, and generate clean bounding-box proposals for downstream tracking and behavioral analysis.

---

## Overview

The pipeline processes each input frame through the following stages:

```
Input Video
     │
     ▼
Background Subtraction (MOG2)
     │
     ▼
Motion Mask Generation
     │
     ▼
Morphological Noise Removal
     │
     ▼
Contour Detection
     │
     ▼
ROI (Bounding Box) Extraction
     │
     ▼
Static Region Exclusion
     │
     ▼
Camera Shake Compensation
     │
     ▼
Final ROIs
```

The resulting ROIs are used as input for the tracking module.

---

## Features

### Motion Detection

- Background subtraction using OpenCV's MOG2
- Shadow-aware foreground detection
- Configurable background learning rate
- Motion mask generation for every frame

### ROI Detection

- Morphological opening and closing for noise suppression
- Contour extraction
- Minimum-area filtering
- Bounding-box generation
- Standardized ROI interface for downstream modules

### Robustness Improvements

- Static exclusion masks for persistent background motion
- Optional ignore-region support
- Phase-correlation based camera shake compensation
- Configurable stabilization pipeline

### Evaluation & Benchmarking

- Frame Difference baseline
- MOG2-only baseline
- ROI pipeline evaluation
- Full pipeline evaluation
- Pixel-level Precision, Recall and F1 evaluation for motion masks
- Box-level IoU-based Precision, Recall and F1 evaluation for ROI outputs
- CSV benchmark generation for quantitative comparison

---

## Repository Structure

```
src/
└── motion/
    ├── motion.py
    ├── roi.py
    ├── shake_compensation.py
    ├── exclusion_regions.py
    ├── baseline_benchmark.py
    ├── test_motion.py
    └── test_roi.py
```

---

## Pipeline Components

### `motion.py`

Implements foreground extraction using MOG2 background subtraction and provides the primary motion detection interface.

**Output**

```python
get_motion_mask(frame) -> binary_mask
```

---

### `roi.py`

Processes motion masks to generate clean Regions of Interest.

Operations include:

- Morphological filtering
- Contour extraction
- Area filtering
- Bounding-box generation

**Output**

```python
get_rois(mask) -> List[(x1, y1, x2, y2)]
```

---

### `shake_compensation.py`

Reduces false detections caused by camera vibration using phase-correlation based frame alignment before motion detection.

---

### `exclusion_regions.py`

Defines static regions that should be ignored during ROI generation (for example, persistent background motion such as ceiling fans).

---

### `baseline_benchmark.py`

Runs quantitative evaluation across multiple pipeline variants and generates benchmark logs.

Evaluated variants:

- Frame Difference
- MOG2 Only
- MOG2 + ROI
- Full Pipeline

Generated metrics:

- Precision
- Recall
- F1 Score
- IoU-based ROI evaluation
- CSV benchmark reports

---

## Dataset

Development and evaluation were performed using selected sequences from the **CDNet2014** dataset, covering multiple challenging scenarios including:

- Shadow
- Dynamic Background
- Low Frame Rate
- Camera Jitter

---

## Design Goals

The module is designed to:

- Detect meaningful motion reliably
- Minimize background noise and false detections
- Produce stable ROI proposals
- Integrate seamlessly with downstream tracking
- Support reproducible quantitative evaluation

---

## Output

For each processed frame, the module provides:

- Binary foreground mask
- Clean Regions of Interest (bounding boxes)
- Visualization overlays
- Benchmark metrics
- CSV evaluation reports

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- SciPy
- Matplotlib