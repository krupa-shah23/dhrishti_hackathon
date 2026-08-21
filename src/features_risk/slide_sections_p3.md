# P3 Slide Sections — Events & Behavior

## 1. Feature Importance Methodology
We extract 11 features per event spanning four modalities:
- **Motion**: motion_area, roi_size, density, duration
- **Behavioral**: frequency (repetition_count), mog2_foreground_ratio
- **Audio**: audio_energy, onset_strength (librosa RMS + spectral onset)
- **Contextual**: object_flag (P2 detection), phase_start/phase_end (exam-phase timing)

Feature importance is computed via XGBoost's built-in gain-based ranking — showing which
signals the model relies on most to separate Normal from Suspicious events. This gives
investigators an interpretable answer to "why was this flagged," not a black-box score.

**Methodology caveat (stated honestly):** current importance rankings are measured on
synthetic bootstrap data; real importance will be re-measured once human-labeled real
events replace the placeholder labels — same model, same features, real data.

## 2. Severity Scoring Architecture
- **Model**: XGBClassifier (gradient-boosted decision trees)
- **Config**: n_estimators=200, max_depth=4, learning_rate=0.05
- **Why these params**:
  - max_depth=4 — keeps individual trees shallow, reducing overfitting risk on a small
    (40-60 event) labeled dataset
  - learning_rate=0.05 — small, careful per-tree corrections rather than aggressive
    single-tree fitting, improving generalization on limited data
  - n_estimators=200 — enough trees to build a stable ensemble without needing GPU
    acceleration; trains in seconds on CPU
- **Output**: severity_score = P(Suspicious | event features), a continuous 0-1
  probability, not a binary flag — lets investigators triage by confidence level
- **Train/test split**: 80/20 stratified, preserving class balance between splits

## 3. Explainability — Real Examples
Generated directly from `explain.py` on real feature output:
- "Seat Desk9: motion duration 0.69s, intensity 22.3, object detected: paper"
- "Seat Desk1: motion duration 4.29s, intensity 19.9, object detected: phone"
- "Seat Desk5: motion duration 3.05s, intensity 95.7, object detected: phone"

Every flagged event ships with a plain-language justification — not just a red box —
directly addressing the plan's explainability requirement and building investigator trust.

## 4. Known Limitations (P3 Scope)
- Ground-truth labels currently synthetic/bootstrap — real F1 pending human-labeled
  organiser footage (biggest open item, actively being worked)
- Severity weights not yet tuned against real repetition patterns (clip 3) — generic
  config until real labels arrive
- Audio corroboration untested on real organiser clips — audio-track availability
  unconfirmed for the dataset
- Phase weighting (start/mid/end) built and verified against real video duration, but
  not yet validated against real behavioral patterns at each phase
- Segmentation/clustering built and tested only against mock motion signals — pending
  P1's real fused signal for final validation