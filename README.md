# Motion & ROI Detection Module

Detects meaningful motion in surveillance/exam-hall footage, suppresses background and camera noise, and produces clean bounding-box proposals for downstream tracking.

---

## Pipeline

```
Input (image sequence or .avi video)
     │
     ▼
Background Subtraction (MOG2, shadow-aware)
     │
     ▼
Morphological Noise Removal (open → close → dilate)
     │
     ▼
Static Exclusion Mask  (per-clip validated background regions)
     │
     ▼
Contour Extraction → Area / Aspect / Fill-ratio Filtering
     │
     ▼
Full-Frame Blob Rejection  (MAX_AREA_FRACTION)
     │
     ▼
Final ROIs  →  [(x1, y1, x2, y2), ...]
```

Camera-shake compensation (phase correlation) is available as an optional pre-step — see **Known Limitations** before enabling it.

---

## Repository Structure

```
src/
└── motion/
    ├── motion.py               # MotionEstimator, get_motion_mask(), video + image-sequence loading
    ├── roi.py                  # get_rois(), exclusion mask + full-frame blob filtering
    ├── shake_compensation.py   # FrameStabilizer (phase-correlation)
    ├── exclusion_regions.py    # per-clip validated static exclusion regions
    ├── export_rois.py          # CSV export for downstream/offline use
    ├── eval_ground_truth.py    # pixel + box-level P/R/F1 vs CDNet/OEP ground truth
    ├── baseline_benchmark.py   # frame-diff / MOG2-only / full-pipeline comparison
    ├── scan_fullframe_blobs.py # sanity sweep for full-frame false positives
    ├── spot_check_min_area.py  # min-area threshold sweep + visual validation
    ├── test_motion.py
    └── test_roi.py             # main benchmark/comparison driver
```

---

## Core Interfaces

```python
get_motion_mask(frame) -> binary_mask          # motion.py
get_rois(mask, exclusion_regions=None) -> [(x1, y1, x2, y2), ...]   # roi.py
```

Box format is locked as `(x1, y1, x2, y2)` across all modules and CSV outputs — **not** `(x, y, w, h)`.

---

## Datasets

- **CDNet2014** — shadow, dynamicBackground, lowFramerate, and cameraJitter categories (9 clips)
- **ShanghaiTech** — 2 clips used for additional pipeline stress-testing
- **MSU OEP** — real exam-proctoring webcam video (`.avi`, 25fps, 640×480, 24 subjects). Only webcam files are used (wearcam is head-mounted, incompatible with static-camera MOG2). Requires `OEP_MIN_AREA=1500` (vs. CDNet defaults) due to close-up-framing texture noise.

---

## Evaluation

- `test_roi.py` — before/after comparisons for exclusion masks and stabilization, plus a standard run across all clips
- `eval_ground_truth.py` — pixel- and IoU-matched box-level Precision/Recall/F1 against ground truth (uses `baseline_benchmark.py`'s box-matching, not raw rasterization)
- `baseline_benchmark.py` — quantifies the incremental value of morphology/contour ROI logic over frame-diff-only and MOG2-only baselines

---

## Technologies

Python 3, OpenCV, NumPy, SciPy, Matplotlib