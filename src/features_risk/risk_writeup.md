# Risk Model Summary (P3 — Features & Risk Model)

## Approach
- Extracted 14 per-event features: motion_area, duration, frequency, speed,
  roi_size, direction, density, persistence, audio_energy, onset_strength,
  object_flag, invigilator_flag, phase_start, phase_end.
- Combined motion (P1) + tracking/detection (P2) + audio (librosa) + exam-phase
  timing (P4) signals per event.
- Trained XGBClassifier (n_estimators=200, max_depth=4, learning_rate=0.05) on
  labeled events (Normal/Suspicious), 80/20 stratified split.

## Feature Importance (on current synthetic data)
1. object_flag (phone/paper detected) — dominant signal (~0.58–0.74)
2. speed — fast movement correlates with suspicious activity
3. frequency — repeated motion bursts
4. audio_energy / onset_strength, invigilator_flag, phase_start/end — near-zero
   currently, since synthetic labels weren't derived from these signals

## Model Status — IMPORTANT
Current Accuracy/Precision/Recall/F1 = 1.000 is a **synthetic pipeline sanity
check only**, NOT a real performance metric. Labels were formula-derived from
the same features the model trains on (speed, frequency, object_flag), so the
model is re-deriving that formula rather than learning genuine suspicious-vs-
normal patterns. This confirms the pipeline (extract → label → train → score)
runs correctly end-to-end; it says nothing about real-world accuracy.

Real evaluation pending:
- P1 confirmed ShanghaiTech clips (01_001, 01_0015) as cross-scene real footage,
  already run through her ROI pipeline.
- P2 to generate real tracks from these ROI outputs.
- P3 to run validate_feature_drift.py on real tracks, then manually label
  exam-relevant Normal/Suspicious events (ShanghaiTech has no native exam
  labels — general surveillance footage only, useful for feature-distribution
  and pipeline validation, not exam-behavior ground truth).
- Exam-hall footage (separate source) will provide the actual labeled events
  for final risk-model evaluation.

Expected real F1 once retrained on real labels: realistic 0.6–0.85 range, not 1.0.

## Ablation: Audio Fusion / Invigilator-Filter / Exam-Phase
| Config | F1 |
|---|---|
| Baseline (no audio/invigilator/phase) | 1.000 |
| + audio-fusion | 1.000 |
| + invigilator-filter | 1.000 |
| + exam-phase-logic | 1.000 |
| All combined | 1.000 |

Flat F1 across all configs is expected by construction on synthetic labels —
none of these signals were used to generate the labels, so they can't move F1
here. Not evidence these features are useless; real ablation gains are only
measurable once real human-labeled events exist.

## Detector Reliability — Real Finding (from P2's FP/FN sweep)
22 FN, 7 FP (5 effective after 2 flagged as likely GT gaps, not detector faults).
All FN are phone-class, concentrated in active-copying windows — object_flag alone
under-detects during genuine Suspicious events. Mitigation: added object_flag_soft
(confidence-weighted) alongside binary flag, and rely more heavily on motion/duration
features to catch events even when detection misses. FP confidences all <0.5 —
threshold at 0.5 for binary flag is empirically justified by this data.

clip3's 8 events now use real detect_objects() output (not ROI-cropped, full-frame inference) — 3/8 windows show real phone detections (0.41-0.52 confidence), 4/8 are confirmed real misses (matches known FN pattern from Day 4 sweep), 1/8 correctly empty (phone genuinely hidden). Other clips still placeholder pending P2's remaining runs.

## Limitations
- Current results validated on synthetic/bootstrapped labels only.
- Small labeled dataset (~80 events) risks overfitting — kept model shallow
  (max_depth=4) to mitigate.
- Audio signal degrades with background noise (fans, chatter) — visual features
  remain primary; audio is a corroborating/fallback signal only.
- ShanghaiTech provides real tracks but not exam-specific labels — must be
  manually labeled for domain-relevant evaluation.

## Next Steps
1. Run validate_feature_drift.py once P2 delivers real ShanghaiTech tracks.
2. Manually label real events (Normal/Suspicious) — exam-relevant criteria,
   not formula-derived.
3. Retrain XGBoost on real labels; report realistic F1 (target >0.7).
4. Re-run full ablation table (audio-fusion, invigilator-filter, exam-phase)
   on real data.
5. Hand off risk_score(features) + confirmed 14-column schema to P4 for
   main.py integration and CSV logging.