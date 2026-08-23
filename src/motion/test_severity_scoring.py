"""
test_severity_scoring.py
Unit tests for src/motion/severity_scoring.py (ML master doc §3.7)

Covers:
  1. Normal case (all inputs present)
  2. Zero/None object_confidence — gating rule: score must still be non-zero
  3. Zero motion + duration edge case
  4. Real-pipeline numbers from clip 04 (event_2: duration=3.125s, no object detection)
  5. Custom weights
  6. score_to_risk_label thresholds and confidence band coverage
"""

import unittest

try:
    from src.motion.severity_scoring import (
        compute_severity,
        score_to_risk_label,
        DEFAULT_WEIGHTS,
        NORM_CEIL,
        THRESHOLDS,
    )
except ImportError:
    from motion.severity_scoring import (
        compute_severity,
        score_to_risk_label,
        DEFAULT_WEIGHTS,
        NORM_CEIL,
        THRESHOLDS,
    )


class TestComputeSeverity(unittest.TestCase):

    # -----------------------------------------------------------------------
    # 1. Normal case — all inputs present and meaningful
    # -----------------------------------------------------------------------
    def test_normal_case_all_inputs(self):
        score = compute_severity(
            motion_intensity=0.08,   # moderate motion (>50% of ceil 0.15)
            duration=15.0,           # 15 s (50% of ceil 30 s)
            repetition_count=5,      # 5 detection hits (50% of ceil 10)
            object_confidence=0.75,  # strong detection
        )
        # Expected: ~50% across all four normalised inputs → score ≈ 0.50
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # With symmetric 50% inputs and default weights the score should be ~0.50
        self.assertAlmostEqual(score, 0.5, delta=0.05)

    # -----------------------------------------------------------------------
    # 2a. object_confidence = 0 → score still non-zero (gating rule)
    # -----------------------------------------------------------------------
    def test_zero_object_confidence_does_not_gate(self):
        score_no_obj = compute_severity(
            motion_intensity=0.08,
            duration=15.0,
            repetition_count=5,
            object_confidence=0.0,
        )
        score_with_obj = compute_severity(
            motion_intensity=0.08,
            duration=15.0,
            repetition_count=5,
            object_confidence=0.75,
        )
        # CRITICAL: zero confidence must not suppress the score
        self.assertGreater(score_no_obj, 0.0,
            "severity_score must be > 0 when motion+duration are non-zero, "
            "even if object_confidence = 0 (object detection is a booster, not a gate)")
        # With obj_conf the score should be strictly higher (it's a booster)
        self.assertGreater(score_with_obj, score_no_obj)

    # -----------------------------------------------------------------------
    # 2b. object_confidence = None → treated as 0, same gating rule
    # -----------------------------------------------------------------------
    def test_none_object_confidence_does_not_gate(self):
        score = compute_severity(
            motion_intensity=0.08,
            duration=15.0,
            repetition_count=5,
            object_confidence=None,
        )
        self.assertGreater(score, 0.0,
            "None object_confidence must be treated as 0, not raise or suppress score")
        # Confirm None and 0.0 produce identical results
        score_zero = compute_severity(0.08, 15.0, 5, 0.0)
        self.assertEqual(score, score_zero)

    # -----------------------------------------------------------------------
    # 3. Zero motion + zero duration → score = 0.0 (no signal at all)
    # -----------------------------------------------------------------------
    def test_zero_motion_and_duration_gives_zero(self):
        score = compute_severity(
            motion_intensity=0.0,
            duration=0.0,
            repetition_count=0,
            object_confidence=None,
        )
        self.assertEqual(score, 0.0,
            "All-zero inputs must produce 0.0 severity score")

    def test_zero_motion_and_duration_with_object_confidence(self):
        # Even if object detection fires, zero motion+duration → score driven
        # only by the object booster (w4=0.10 of max)
        score = compute_severity(
            motion_intensity=0.0,
            duration=0.0,
            repetition_count=0,
            object_confidence=1.0,
        )
        # w4 = 0.10, total_w = 1.0 → score = 0.10
        self.assertAlmostEqual(score, DEFAULT_WEIGHTS["w4"], delta=0.001,
            msg="With zero motion+duration, score should equal w4 * norm_obj_conf")

    # -----------------------------------------------------------------------
    # 4. REAL PIPELINE NUMBERS from clip 04, event_2
    #    (extracted 2026-08-21 by extract_real_events.py on first 50 frames)
    #    event_2: duration=3.125s, total_frames=26, object_detected=False,
    #             repetition_count=0, max_obj_confidence=None
    #    motion_intensity: approximate from MOG2 on clip 04 (~3-6% pixel activity
    #    in an exam hall with ~6-10 students moving slightly)
    # -----------------------------------------------------------------------
    def test_real_event_clip04_event2(self):
        # Conservative real-world motion: ~4% pixel activity in exam hall
        score = compute_severity(
            motion_intensity=0.04,   # ~4% pixels flagged — real exam-hall level
            duration=3.125,          # actual event_2 duration from pipeline output
            repetition_count=0,      # no object detection (weights not available)
            object_confidence=None,  # no detections
        )
        # Expected: low-medium. duration=3.125/30=~10%, motion=0.04/0.15=~27%
        # Weighted: 0.40*0.267 + 0.35*0.104 + 0 + 0 = 0.107 + 0.036 = 0.143
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.30,
            "Short, low-motion real event should score well below 0.30")
        # The score should be in the low band → yellow label
        risk_label, color_tag, confidence = score_to_risk_label(score)
        self.assertEqual(risk_label, "low")
        self.assertEqual(color_tag, "yellow")

    def test_real_event_clip04_sustained_hypothetical(self):
        # What if event_6 (3.625s) had moderate motion + 2 detection hits?
        score = compute_severity(
            motion_intensity=0.06,
            duration=3.625,
            repetition_count=2,
            object_confidence=0.55,
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # -----------------------------------------------------------------------
    # 5. Custom weights — verify override works and sum-normalisation holds
    # -----------------------------------------------------------------------
    def test_custom_weights_fully_motion_driven(self):
        # If w1=1, others=0, score should equal norm_motion alone
        score = compute_severity(
            motion_intensity=0.075,  # 50% of ceil (0.15)
            duration=30.0,
            repetition_count=10,
            object_confidence=1.0,
            weights={"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0},
        )
        self.assertAlmostEqual(score, 0.5, delta=0.001)

    def test_custom_weights_zero_all_raises_no_error(self):
        score = compute_severity(
            motion_intensity=0.05,
            duration=10.0,
            repetition_count=3,
            object_confidence=0.6,
            weights={"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0},
        )
        self.assertEqual(score, 0.0)

    # -----------------------------------------------------------------------
    # 6. score_to_risk_label — threshold and confidence band coverage
    # -----------------------------------------------------------------------
    def test_low_band(self):
        risk_label, color_tag, confidence = score_to_risk_label(0.0)
        self.assertEqual(risk_label, "low")
        self.assertEqual(color_tag, "yellow")
        self.assertEqual(confidence, 0)

    def test_low_band_midpoint(self):
        risk_label, color_tag, confidence = score_to_risk_label(0.124)
        self.assertEqual(risk_label, "low")
        self.assertEqual(color_tag, "yellow")
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 50)

    def test_medium_band(self):
        risk_label, color_tag, confidence = score_to_risk_label(0.25)
        self.assertEqual(risk_label, "medium")
        self.assertEqual(color_tag, "yellow")
        self.assertGreaterEqual(confidence, 50)
        self.assertLess(confidence, 75)

    def test_high_band(self):
        risk_label, color_tag, confidence = score_to_risk_label(0.50)
        self.assertEqual(risk_label, "high")
        self.assertEqual(color_tag, "red")
        self.assertGreaterEqual(confidence, 75)

    def test_max_score(self):
        risk_label, color_tag, confidence = score_to_risk_label(1.0)
        self.assertEqual(risk_label, "high")
        self.assertEqual(color_tag, "red")
        self.assertEqual(confidence, 100)

    def test_score_clamped_above_one(self):
        # Defensive: callers should never pass > 1.0, but function must not crash
        risk_label, color_tag, confidence = score_to_risk_label(2.5)
        self.assertEqual(risk_label, "high")
        self.assertEqual(color_tag, "red")
        self.assertLessEqual(confidence, 100)

    def test_score_clamped_below_zero(self):
        risk_label, color_tag, confidence = score_to_risk_label(-0.5)
        self.assertEqual(risk_label, "low")
        self.assertEqual(color_tag, "yellow")
        self.assertGreaterEqual(confidence, 0)

    def test_confidence_is_int(self):
        _, _, confidence = score_to_risk_label(0.37)
        self.assertIsInstance(confidence, int)

    # -----------------------------------------------------------------------
    # 7. Output bounds — score always in [0, 1]
    # -----------------------------------------------------------------------
    def test_score_is_bounded(self):
        for motion in [0.0, 0.05, 0.15, 1.0]:
            for dur in [0.0, 5.0, 30.0, 300.0]:
                for rep in [0, 5, 50]:
                    for conf in [None, 0.0, 0.5, 1.0]:
                        score = compute_severity(motion, dur, rep, conf)
                        self.assertGreaterEqual(score, 0.0, f"Below 0: {motion},{dur},{rep},{conf}")
                        self.assertLessEqual(score, 1.0, f"Above 1: {motion},{dur},{rep},{conf}")


if __name__ == "__main__":
    unittest.main()
