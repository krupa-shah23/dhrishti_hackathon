# Product Requirements Document (PRD)
## Offline Exam-Hall Video Analytics — P4 (Outputs, Evaluation & Integration)

---

## 1. Executive Summary & Problem Statement Alignment

### Problem Statement #2: Offline Video Segmentation and ROI Detection Using Motion Estimation
Examination authorities often need to review extensive recorded surveillance footage post-examinations to investigate reported incidents, verify suspicious behavior, and audit overall exam integrity. Manual review of multi-hour video feeds across dozens of examination halls is labor-intensive, error-prone, and inefficient.

### Proposed Solution
An AI-powered offline video analytics system that automatically processes recorded examination footage to:
1. **Detect & Localize Motion ROIs**: Identify frame-to-frame motion patterns, localizing Regions of Interest (ROI) containing relevant activity while filtering out static background noise.
2. **Segment Video Footage into Events**: Group temporal motion spikes into discrete activity clips for targeted human review.
3. **Detect Prohibited Objects & Track Student Actions**: Locate objects like mobile phones or paper chits, tracking movement across frames.
4. **Generate Explainable Analytics & Summaries**: Produce motion heatmaps, activity timeline graphs, searchable CSV event logs, and automated PDF summary reports.

---

## 2. Architecture Pipeline Validation

The team has constructed a 4-Module Decoupled Architecture (P1 through P4):

```
       ┌─────────────────────────────────────────────────────────┐
       │                Recorded Video Stream                    │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ P1: Motion & ROI Detection (src/motion/)                               │
 │ - MOG2 Background Subtraction, Optical Flow, Contours & Morphology      │
 │ - Contract: get_motion_mask(frame) -> mask                             │
 │ - Contract: get_rois(mask) -> [boxes]                                  │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ P2: Tracking & Object Detection (src/track_det/)                       │
 │ - ByteTrack / Centroid Tracking & YOLO Object Classifier                │
 │ - Contract: track(boxes) -> [tracks]                                   │
 │ - Contract: detect_objects(frame) -> [boxes, class, conf]              │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ P3: Feature Extraction & Risk Scoring (src/features_risk/)            │
 │ - Spatial-Temporal Movement Features & Audio Energy Fusion              │
 │ - Contract: extract_features(track) -> dict                            │
 │ - Contract: risk_score(features) -> float                              │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ P4: Outputs, Evaluation & Integration (src/outputs_eval/ + main.py)    │
 │ - Orchestration Pipeline (main.py)                                     │
 │ - Heatmap Generation (heatmap.py) & Motion Timeline (timeline.py)       │
 │ - CSV Event Logger (logger.py) & PDF/JSON Summary Report (report.py)   │
 │ - Quantitative Evaluation (metrics.py, benchmark.py, ablation.py)     │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Pipeline Validation Analysis
- **Correctness & Fit**: **YES**, the pipeline is modular, robust, and directly satisfies Problem Statement #2.
  - **P1** filters out 80%+ static background frames, solving computational efficiency for long recordings.
  - **P2** tracks entities continuously and detects prohibited objects (phones/chits).
  - **P3** computes behavioral risk scores to separate natural movements (stretching/looking up) from suspicious events.
  - **P4** encapsulates all user-facing deliverables, metrics evaluation, and benchmark analysis.
- **Decoupling**: Standardized Python signatures frozen on Day 1 allow all 4 sub-teams to work in parallel without blocking each other.

---

## 3. P4 Scope, Responsibilities & Key Objectives

As **P4 (Outputs, Evaluation & Integration Lead)**, the core deliverables and responsibilities are:

### Core Responsibilities
1. **Pipeline Orchestrator (`main.py`)**: Integrates modules P1, P2, P3 via frozen function contracts to run end-to-end processing.
2. **Visual & Report Outputs**:
   - **Heatmap Generator (`src/outputs_eval/heatmap.py`)**: Accumulates per-frame motion masks into a cumulative spatial matrix and overlays color maps (`cv2.COLORMAP_JET`) to highlight examination "hotspots".
   - **Activity Timeline (`src/outputs_eval/timeline.py`)**: Generates temporal plots mapping motion intensity and risk scores against video timestamp.
   - **CSV Event Logger (`src/outputs_eval/logger.py`)**: Records structured event logs (`timestamp`, `ROI_box`, `motion_score`, `audio_level`, `risk_score`, `confidence`).
   - **PDF/JSON Summary Report (`src/outputs_eval/report.py`)**: Compiles comprehensive executive summaries for invigilators/investigators.
3. **Quantitative Evaluation & Benchmarking**:
   - **Metrics Engine (`src/outputs_eval/metrics.py`)**: Computes IoU, Precision, Recall, and F1-score (with temporal overlap criterion $\ge 50\%$).
   - **Benchmark Runner (`src/outputs_eval/benchmark.py`)**: Compares Frame-Diff vs. MOG2 vs. MOG2+ROI vs. Full Pipeline.
   - **Ablation Studies (`src/outputs_eval/ablation.py`)**: Measures F1-score impact of removing Invigilator Filter, Exam Phase Logic, or Audio Fusion.

---

## 4. Frozen Day-1 Function Contracts

```python
# P1 — Motion & ROI
get_motion_mask(frame: np.ndarray) -> np.ndarray
get_rois(mask: np.ndarray) -> List[Tuple[int, int, int, int]]

# P2 — Tracking & Object Detection
track(boxes: List[Tuple[int, int, int, int]]) -> List[Dict]
detect_objects(frame: np.ndarray) -> List[Tuple[List[int], str, float]]

# P3 — Feature Extraction & Risk Scoring
extract_features(track: Dict) -> Dict
risk_score(features: Dict) -> float
```

---

## 5. Evaluation Metrics & Performance Benchmarks

| Metric | Target | Description |
|---|---|---|
| **Event Detection F1-Score** | $\ge 0.75$ | Evaluated with $\ge 50\%$ temporal overlap on annotated evaluation clips |
| **Mask IoU (CDnet2014)** | $\ge 0.65$ | Spatial motion overlap accuracy |
| **Processing Speed** | $\ge 25$ FPS | Real-time / Faster-than-real-time batch throughput on GPU/CPU |
| **False Event Reduction** | $\ge 40\%$ | Reduction in flagged events compared to naive frame-differencing baseline |

---

## 6. Risk Factors & Edge Case Mitigations

| Edge Case | Potential Impact | Mitigation Strategy (P4 Architecture) |
|---|---|---|
| **Multi-hour Videos** | High memory usage, processing bottlenecks | Chunk-wise frame processing & stream generator loops |
| **Lighting Changes / Camera Noise** | False motion spikes across full frame | Frame normalization & MOG2 history parameter tuning |
| **Over/Under-Segmentation** | Fragmented or merged event logs | Hysteresis temporal thresholding (min duration windowing) |
| **Data Storage Limits** | Huge storage consumption for raw masks | Stream and compress raw masks; retain only CSV + PNG heatmaps |

---

## 7. Deliverables Checklist

- [x] Frozen Day-1 Function Contracts (`docs/function_contracts.md`)
- [ ] End-to-end Video Pipeline Orchestration (`main.py`)
- [ ] Motion Heatmap Overlay (`src/outputs_eval/heatmap.py`)
- [ ] Motion Score Timeline Plot (`src/outputs_eval/timeline.py`)
- [ ] CSV Event Logger (`src/outputs_eval/logger.py`)
- [ ] PDF Summary Report Generator (`src/outputs_eval/report.py`)
- [ ] Metric Calculation Engine (`src/outputs_eval/metrics.py`)
- [ ] Benchmark & Ablation Study Suite (`src/outputs_eval/benchmark.py`, `src/outputs_eval/ablation.py`)
