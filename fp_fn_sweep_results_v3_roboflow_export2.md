# Evaluation Metadata: `fp_fn_sweep_results_v3_roboflow_export2.csv`

- **Model**: `models/phone_detector_v3.pt`
- **Evaluation Dataset**: `data/roboflow_export2` (831 images total: 702 train / 99 valid / 30 test)
- **Evaluation Date**: 2026-08-22
- **Summary Metrics**:
  - **Phone**: 210 Ground-Truth Images | 197 Detected (TP) | 13 Missed (FN) | 25 False Alarms (FP) — **93.8% Recall**
  - **Paper-Chit**: 113 Ground-Truth Images | 96 Detected (TP) | 17 Missed (FN) | 0 False Alarms (FP) — **85.0% Recall**
  - **Total Overall Errors**: 30 FN, 25 FP

## Critical Caveat on Comparability
> [!WARNING]
> **NOT DIRECTLY COMPARABLE TO THE OLD 22 FN / 7 FP v1 BASELINE:**
> The earlier 22 FN / 7 FP baseline was computed on the older 593-image dataset (`data/roboflow_export`). This evaluation was performed on the updated 831-image dataset (`data/roboflow_export2`, +238 new images). 
> 
> The change from 22 FN to 30 FN is **NOT a regression**:
> 1. The total evaluation image count increased by 40% (593 → 831).
> 2. The older v1 baseline had 22 phone FNs on 593 images. On this 831-image set, v3 has only 13 phone FNs across the entire 831 images (a significant increase in phone recall).
> 3. Paper-chit accounts for 17 of the 30 FNs, with 96 out of 113 total paper-chit instances successfully detected (85.0% recall).
