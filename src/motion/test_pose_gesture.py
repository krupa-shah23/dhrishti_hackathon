"""
test_pose_gesture.py
Unit tests for P1 pose/gesture signal module (ML master doc §3.6).

Tests cover all four sub-signals:
    1. Sustained gaze-shift
    2. Chit-passing (wrist proximity)
    3. Targeted scanning
    4. End-to-end activities[] wiring through P2P3Bridge

Each test is self-contained and runs without real video / without MediaPipe
installed, by driving the internal state machines directly.
"""

import math
import collections
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ---------------------------------------------------------------------------
# Make tests runnable both as `python -m unittest` from repo root AND as
# `python test_pose_gesture.py` from within src/motion/
# ---------------------------------------------------------------------------
try:
    from src.motion.pose_gesture import (
        PoseGestureAnalyzer,
        _TrackPoseState,
        _WristState,
        _angle_to_octant,
        GAZE_ANGLE_THRESHOLD_DEG,
        GAZE_SUSTAIN_SEC,
        WRIST_PROXIMITY_PX,
        SCAN_MIN_SNAPS,
        SNAP_MIN_DELTA_DEG,
        integrate_pose_into_event,
        signals_to_activities,
    )
    from src.integration.p2_p3_bridge import P2P3Bridge
except ImportError:
    from pose_gesture import (
        PoseGestureAnalyzer,
        _TrackPoseState,
        _WristState,
        _angle_to_octant,
        GAZE_ANGLE_THRESHOLD_DEG,
        GAZE_SUSTAIN_SEC,
        WRIST_PROXIMITY_PX,
        SCAN_MIN_SNAPS,
        SNAP_MIN_DELTA_DEG,
        integrate_pose_into_event,
        signals_to_activities,
    )
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from integration.p2_p3_bridge import P2P3Bridge


# ---------------------------------------------------------------------------
# Sub-signal 1: Sustained gaze-shift
# ---------------------------------------------------------------------------

class TestSustainedGazeShift(unittest.TestCase):
    """_TrackPoseState.update_gaze() must:
    - return False while head is forward or angle < threshold
    - return False while shifted but < GAZE_SUSTAIN_SEC
    - return True at exactly GAZE_SUSTAIN_SEC of continuous shift
    - reset timer when head returns to neutral
    """

    def setUp(self):
        self.state = _TrackPoseState(fps=25.0)

    def test_no_signal_when_forward(self):
        for t in range(10):
            result = self.state.update_gaze(yaw_deg=0.0, pitch_deg=0.0, ts=float(t))
        self.assertFalse(result)

    def test_no_signal_before_sustain_threshold(self):
        # Yaw above threshold but only held for 4 seconds
        yaw = GAZE_ANGLE_THRESHOLD_DEG + 10.0
        results = [self.state.update_gaze(yaw, 0.0, float(t)) for t in range(4)]
        self.assertFalse(any(results), "Should not fire before 5s sustained")

    def test_signal_fires_at_sustain_threshold(self):
        yaw = GAZE_ANGLE_THRESHOLD_DEG + 10.0
        results = []
        for t in range(int(GAZE_SUSTAIN_SEC) + 1):
            results.append(self.state.update_gaze(yaw, 0.0, float(t)))
        self.assertTrue(any(results), "Should fire after GAZE_SUSTAIN_SEC seconds")

    def test_timer_resets_on_neutral(self):
        yaw = GAZE_ANGLE_THRESHOLD_DEG + 10.0
        # Shift for 4s, return to neutral, shift again — should NOT fire at t=9
        for t in range(4):
            self.state.update_gaze(yaw, 0.0, float(t))
        self.state.update_gaze(0.0, 0.0, 4.0)  # neutral resets timer
        results = [self.state.update_gaze(yaw, 0.0, float(t)) for t in range(5, 9)]
        # only 4 more seconds — below threshold again
        self.assertFalse(any(results), "Timer should have reset on neutral")

    def test_pitch_also_triggers(self):
        pitch = GAZE_ANGLE_THRESHOLD_DEG + 5.0
        results = []
        for t in range(int(GAZE_SUSTAIN_SEC) + 1):
            results.append(self.state.update_gaze(0.0, pitch, float(t)))
        self.assertTrue(any(results), "Pitch > threshold should also trigger sustained_gaze_shift")


# ---------------------------------------------------------------------------
# Sub-signal 2: Chit-passing (wrist proximity)
# ---------------------------------------------------------------------------

class TestChitPassing(unittest.TestCase):
    """_WristState.check_proximity() must:
    - return [] when no pair is within WRIST_PROXIMITY_PX
    - return [(tid_a, tid_b)] when wrists are close
    - ignore tracks with unknown/invisible wrists
    """

    def setUp(self):
        self.ws = _WristState()

    def test_no_hit_when_far(self):
        # Two tracks, wrists separated by >> threshold
        self.ws.update(1, (100.0, 100.0), None)
        self.ws.update(2, (300.0, 300.0), None)
        self.assertEqual(self.ws.check_proximity(), [])

    def test_hit_when_close(self):
        # Right wrist of track 1 near left wrist of track 2
        self.ws.update(1, None, (200.0, 200.0))
        self.ws.update(2, (200.0 + WRIST_PROXIMITY_PX - 1.0, 200.0), None)
        hits = self.ws.check_proximity()
        self.assertTrue(len(hits) > 0, "Expected chit-passing hit")
        self.assertIn(1, hits[0])
        self.assertIn(2, hits[0])

    def test_no_hit_when_just_over_threshold(self):
        # Just over threshold by 1px — should NOT trigger
        self.ws.update(1, None, (0.0, 0.0))
        self.ws.update(2, (WRIST_PROXIMITY_PX + 1.0, 0.0), None)
        hits = self.ws.check_proximity()
        self.assertEqual(hits, [], "Just over threshold should NOT trigger")

    def test_no_hit_with_single_track(self):
        self.ws.update(1, (100.0, 100.0), (102.0, 102.0))
        self.assertEqual(self.ws.check_proximity(), [])

    def test_evict_removes_track(self):
        self.ws.update(1, (100.0, 100.0), None)
        self.ws.update(2, (101.0, 100.0), None)
        self.ws.evict(2)
        self.assertEqual(self.ws.check_proximity(), [])


# ---------------------------------------------------------------------------
# Sub-signal 3: Targeted scanning
# ---------------------------------------------------------------------------

class TestTargetedScanning(unittest.TestCase):
    """_TrackPoseState.update_scan() must:
    - return False on random non-repeating direction changes
    - return True when SCAN_MIN_SNAPS identical octant snaps occur in window
    - respect SNAP_MIN_DELTA_DEG — micro-movements should not accumulate
    - prune entries outside SCAN_WINDOW_SEC
    """

    def setUp(self):
        self.state = _TrackPoseState(fps=25.0)

    def _snap(self, from_deg, to_deg, ts):
        """Simulate two calls: one at from_deg, one at to_deg (creates a snap)."""
        self.state.update_scan(from_deg, ts)
        return self.state.update_scan(to_deg, ts + 0.1)

    def test_no_signal_random_directions(self):
        # Random octant each call — no single octant repeats 3x
        directions = [0, 45, 90, 135, 180, 225, 270]
        ts = 0.0
        results = []
        for d in directions:
            results.append(self.state.update_scan(float(d), ts))
            ts += 2.0
        self.assertFalse(any(results), "Random scanning should not trigger")

    def test_signal_fires_on_repeated_snaps(self):
        # Snap to octant 2 (~90°) three times from forward (0°)
        ts = 0.0
        fired = False
        for _ in range(SCAN_MIN_SNAPS):
            self.state.update_scan(0.0, ts)
            ts += 1.0
            if self.state.update_scan(90.0, ts):
                fired = True
            ts += 1.0
        self.assertTrue(fired, f"Expected targeted_scanning after {SCAN_MIN_SNAPS} identical snaps")

    def test_micro_movements_ignored(self):
        # Small back-and-forth below SNAP_MIN_DELTA_DEG
        ts = 0.0
        results = []
        for _ in range(20):
            results.append(self.state.update_scan(5.0, ts))
            ts += 0.5
            results.append(self.state.update_scan(5.0 + SNAP_MIN_DELTA_DEG * 0.5, ts))
            ts += 0.5
        self.assertFalse(any(results), "Sub-threshold movements should not trigger")

    def test_window_pruning(self):
        # Put 2 snaps in, then advance time past SCAN_WINDOW_SEC, then 1 more snap
        # Should not fire because old snaps are pruned
        ts = 0.0
        from src.motion.pose_gesture import SCAN_WINDOW_SEC
        self.state.update_scan(0.0, ts)
        self.state.update_scan(90.0, ts + 1.0)
        self.state.update_scan(0.0, ts + 2.0)
        self.state.update_scan(90.0, ts + 3.0)
        # Jump far into future — old snaps should be pruned
        ts = SCAN_WINDOW_SEC + 100.0
        self.state.update_scan(0.0, ts)
        result = self.state.update_scan(90.0, ts + 1.0)
        # Only 1 snap in the new window; should NOT fire
        self.assertFalse(result, "Snaps outside the window should be pruned")


# ---------------------------------------------------------------------------
# Sub-signal 4 end-to-end: activities[] wiring through P2P3Bridge
# ---------------------------------------------------------------------------

class TestActivitiesWiring(unittest.TestCase):
    """End-to-end test: pose signals injected into P2P3Bridge.process_fused_tracks()
    with pose_signals argument must appear in the event's activities[] array
    after the track is finalized.
    """

    def _make_fused_track(self, tid):
        return {
            "track_id": tid,
            "box": (10.0, 10.0, 100.0, 100.0),
            "class": None,
            "confidence": None,
            "invigilator_flag": False,
        }

    def test_pose_signal_reaches_activities(self):
        bridge = P2P3Bridge(missing_threshold=2, fps=25.0)
        ft = self._make_fused_track(tid=1)

        # Frame 0: track appears, pose signal fires
        bridge.process_fused_tracks([ft], frame_index=0,
                                     pose_signals={1: ["sustained_gaze_shift"]})
        # Frame 1: another signal fires
        bridge.process_fused_tracks([ft], frame_index=1,
                                     pose_signals={1: ["targeted_scanning"]})
        # Frame 2,3,4: track disappears → finalized after missing_threshold=2 misses
        for fi in range(2, 5):
            bridge.process_fused_tracks([], frame_index=fi, pose_signals={})

        events = bridge.get_completed_events()
        self.assertEqual(len(events), 1)
        acts = events[0]["activities"]
        self.assertIn("sustained_gaze_shift", acts,
                      f"sustained_gaze_shift missing from activities: {acts}")
        self.assertIn("targeted_scanning", acts,
                      f"targeted_scanning missing from activities: {acts}")

    def test_chit_passing_signal_reaches_activities(self):
        bridge = P2P3Bridge(missing_threshold=2, fps=25.0)
        ft1 = self._make_fused_track(tid=1)
        ft2 = self._make_fused_track(tid=2)

        bridge.process_fused_tracks([ft1, ft2], frame_index=0,
                                     pose_signals={
                                         1: ["chit_passing:1+2"],
                                         2: ["chit_passing:1+2"],
                                     })

        for fi in range(1, 5):
            bridge.process_fused_tracks([], frame_index=fi, pose_signals={})

        events = bridge.get_completed_events()
        all_activities = [a for ev in events for a in ev["activities"]]
        self.assertTrue(
            any("chit_passing" in a for a in all_activities),
            f"chit_passing missing from all events activities: {events}"
        )

    def test_no_pose_signals_gives_empty_activities(self):
        bridge = P2P3Bridge(missing_threshold=2, fps=25.0)
        ft = self._make_fused_track(tid=3)
        bridge.process_fused_tracks([ft], frame_index=0, pose_signals={})
        for fi in range(1, 5):
            bridge.process_fused_tracks([], frame_index=fi, pose_signals={})
        events = bridge.get_completed_events()
        self.assertEqual(events[0]["activities"], [],
                         "No pose signals should leave activities empty")

    def test_duplicate_signals_deduplicated(self):
        bridge = P2P3Bridge(missing_threshold=2, fps=25.0)
        ft = self._make_fused_track(tid=4)
        for fi in range(2):
            bridge.process_fused_tracks([ft], frame_index=fi,
                                         pose_signals={4: ["sustained_gaze_shift"]})
        for fi in range(2, 6):
            bridge.process_fused_tracks([], frame_index=fi, pose_signals={})
        events = bridge.get_completed_events()
        count = events[0]["activities"].count("sustained_gaze_shift")
        self.assertEqual(count, 1, "Duplicate signals should be deduplicated in activities")


# ---------------------------------------------------------------------------
# Helper: signals_to_activities deduplication
# ---------------------------------------------------------------------------

class TestSignalsToActivities(unittest.TestCase):
    def test_dedup_preserves_order(self):
        signals = ["sustained_gaze_shift", "targeted_scanning", "sustained_gaze_shift"]
        result = signals_to_activities(signals)
        self.assertEqual(result, ["sustained_gaze_shift", "targeted_scanning"])

    def test_empty_list(self):
        self.assertEqual(signals_to_activities([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
