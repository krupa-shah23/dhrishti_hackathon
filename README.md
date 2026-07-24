# Offline Exam-Hall Video Analytics — Problem Statement #2

An AI-powered video analytics system that automatically segments recorded examination hall videos and identifies Regions of Interest (ROI) using motion estimation techniques.

## Project Structure & Ownership

- `src/motion/` — **P1 Owner**: Background subtraction (MOG2), optical flow, ROI extraction.
- `src/track_det/` — **P2 Owner**: Object tracking (ByteTrack) & Prohibited Object Detection (YOLO).
- `src/features_risk/` — **P3 Owner**: Feature extraction & XGBoost behavioral risk scoring.
- `src/outputs_eval/` & `main.py` — **P4 Owner**: Orchestration, heatmaps, timelines, CSV logging, PDF reports, & benchmarks.

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements/p4_requirements.txt
   ```
2. Run end-to-end dry integration / demo:
   ```bash
   python main.py
   ```

## Documentation

- [PRD (P4 Focus)](docs/P4_PRD_Outputs_Eval_Integration.md)
- [Function Contracts](docs/function_contracts.md)
- [Status Dashboard](docs/status_sheet.md)
