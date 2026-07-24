# Daily Merge Ritual & Status Sheet

This document tracks daily progress and integration status across all sub-modules during the 5-day execution plan.

---

## Daily Merge Ritual Checklist (~30 min daily)

- [ ] **Step 1**: All team members push latest commits to shared repository.
- [ ] **Step 2**: P4 runs `main.py` end-to-end on fixed benchmark video.
- [ ] **Step 3**: Identify and log function contract breaks or schema mismatches.
- [ ] **Step 4**: Assign fix owners for next morning.
- [ ] **Step 5**: Update status sheet below.

---

## Module Status Dashboard

| Module | Sub-module | Owner | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 |
|---|---|---|---|---|---|---|---|
| **P1** | `mog2_subtractor.py` | P1 | Stub | Pending | Pending | Pending | Pending |
| **P1** | `optical_flow.py` | P1 | Stub | Pending | Pending | Pending | Pending |
| **P1** | `contours_morphology.py` | P1 | Stub | Pending | Pending | Pending | Pending |
| **P1** | `roi_boxes.py` | P1 | Stub | Pending | Pending | Pending | Pending |
| **P2** | `bytetrack_wrapper.py` | P2 | Stub | Pending | Pending | Pending | Pending |
| **P2** | `yolo_infer.py` | P2 | Stub | Pending | Pending | Pending | Pending |
| **P3** | `feature_extraction.py` | P3 | Stub | Pending | Pending | Pending | Pending |
| **P3** | `train_xgboost.py` | P3 | Stub | Pending | Pending | Pending | Pending |
| **P4** | `heatmap.py` | **P4** | Stub | Planned | Integration | Final | Validated |
| **P4** | `timeline.py` | **P4** | Stub | Planned | Integration | Final | Validated |
| **P4** | `logger.py` | **P4** | Stub | Complete | Integration | Final | Validated |
| **P4** | `report.py` | **P4** | Stub | Planned | Integration | Final | Validated |
| **P4** | `metrics.py` | **P4** | Stub | Planned | Integration | Stress Test | Benchmark |
| **P4** | `main.py` | **P4** | Stubs wired | Multi-module | E2E V1 | Full Pipeline | Demo Ready |
