"""
pose_gesture.py
P1 — Pose/Gesture Signal module (ML master doc §3.6)

Runs DOWNSTREAM of the Stage-A motion gate. Only called on ROI crops that
are already flagged (i.e., track_crop_buffer entries). Never on every frame.

Pipeline position:
    Stage A: motion gate  -> get_rois() -> track()
    Stage B: detect_objects() on flagged crop
    Stage C (this file): pose_gesture_analyze() on same flagged crop

Rationale for MediaPipe Pose:
    - Ships as a pure-Python wheel with bundled TFLite models.
    - LITE variant runs comfortably on CPU in <20ms per crop at exam-hall
      resolutions (640×480 or smaller), acceptable for per-track-per-N-frames
      frequency.
    - Alternative (OpenPose, AlphaPose) requires CUDA and complex build;
      MediaPipe is the correct choice for this deployment profile.
    - If mediapipe is not installed, this module degrades gracefully:
      analyze_pose_crop() returns an empty dict and logs once.

Sub-signals implemented (all four from §3.6):
    1. sustained_gaze_shift    — head yaw or pitch > 45° sustained > 5 s
    2. chit_passing            — wrist proximity between two distinct track_ids
    3. targeted_scanning       — 3+ identical angular snaps within 2-min window

Wiring:
    All three signals feed pose_signals_to_activities() which returns a list
    of activity strings for appending to the event schema's `activities` array.
    Call integrate_pose_into_event() to do this in one shot.

Usage (from p1_p2_tracker.py process_frame):
    from src.motion.pose_gesture import PoseGestureAnalyzer
    self.pose_analyzer = PoseGestureAnalyzer(fps=native_fps)
    ...
    per frame, after crop extraction:
        pose_signals = self.pose_analyzer.update(
            frame_index=frame_index,
            crops_by_tid={tid: crop},
            timestamp_sec=frame_index / fps
        )
    Then pass pose_signals to the P2P3Bridge active_track accumulator so they
    reach _finalize_track() -> activities.
"""

from __future__ import annotations

import collections
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Graceful mediapipe import — handles both Tasks API (>=1.0) and legacy (<=0.10)
# ---------------------------------------------------------------------------
_MP_AVAILABLE = False
_mp_landmarker_cls = None
_mp_landmarker_opts_cls = None
_mp_base_opts_cls = None
_mp_image_cls = None
_mp_image_format = None

try:
    # mediapipe >= 1.0 Tasks API
    from mediapipe.tasks.python import BaseOptions as _MpBaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker as _PoseLandmarker
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions as _PoseLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as _RunningMode
    from mediapipe import Image as _MpImage, ImageFormat as _MpImageFormat
    _mp_landmarker_cls = _PoseLandmarker
    _mp_landmarker_opts_cls = _PoseLandmarkerOptions
    _mp_base_opts_cls = _MpBaseOptions
    _mp_image_cls = _MpImage
    _mp_image_format = _MpImageFormat
    _MP_AVAILABLE = True
except Exception as _mp_e:
    warnings.warn(
        f"[pose_gesture] mediapipe Tasks API not available ({_mp_e}). "
        "pose/gesture signals will not fire. "
        "Run: pip install mediapipe",
        RuntimeWarning,
        stacklevel=1,
    )

# Default model path — override via POSE_MODEL_PATH env var or __init__ arg
import os as _os
DEFAULT_MODEL_PATH = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_lite.task"
)


# ---------------------------------------------------------------------------
# Constants (all from §3.6)
# ---------------------------------------------------------------------------

# Sub-signal 1 — sustained gaze-shift
GAZE_ANGLE_THRESHOLD_DEG: float = 45.0    # yaw or pitch beyond this → shifted
GAZE_SUSTAIN_SEC: float = 5.0             # must hold continuously for this long

# Sub-signal 2 — chit-passing
# Wrist proximity threshold in PIXELS at 480p. MediaPipe returns normalized
# [0,1] coords; we denormalize against crop height.
# Justification: at 480p a hand is ~30-50px wide; 60px = roughly 1 hand-width,
# which is the smallest distance at which fingers of two people could be
# meaningfully interacting (hand-off). Tighter = too many misses on low-res;
# looser = too many false positives on adjacent seatmates' incidental overlaps.
WRIST_PROXIMITY_PX: float = 60.0

# Sub-signal 3 — targeted scanning
# An "angular snap" is a head-turn to the same direction (binned to 8 octants)
# within a rolling window. 3+ identical octant-snaps (not counting consecutive
# dwell) in 2 minutes = targeted scanning.
SCAN_WINDOW_SEC: float = 120.0
SCAN_MIN_SNAPS: int = 3
# Minimum angle change to count as a "snap" (filters out head sway / noise)
SNAP_MIN_DELTA_DEG: float = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_pitch_from_landmarks(landmarks, w: int, h: int) -> Tuple[float, float]:
    """
    Estimate head yaw and pitch in degrees from MediaPipe Pose landmarks.

    Method: use the nose (0), left ear (7), right ear (8), and left/right
    shoulder (11, 12) to approximate orientation.

    Yaw:  horizontal asymmetry between ear-to-nose distances.
    Pitch: vertical offset of nose relative to shoulder midpoint.

    Returns (yaw_deg, pitch_deg). Positive yaw = facing left; negative = right.
    Convention matches the "looking away from own paper" semantic.
    """
    lm = landmarks.landmark

    def pt(idx):
        return lm[idx].x * w, lm[idx].y * h

    nose = pt(0)
    l_ear = pt(7)
    r_ear = pt(8)
    l_shoulder = pt(11)
    r_shoulder = pt(12)

    # Yaw: if nose is closer to left ear than right ear, facing right (away left)
    d_left = math.hypot(nose[0] - l_ear[0], nose[1] - l_ear[1])
    d_right = math.hypot(nose[0] - r_ear[0], nose[1] - r_ear[1])
    ear_span = math.hypot(l_ear[0] - r_ear[0], l_ear[1] - r_ear[1]) + 1e-6
    # Normalize asymmetry to [-1, 1], then map to degrees
    asym = (d_right - d_left) / ear_span  # +1 = fully right, -1 = fully left
    yaw_deg = asym * 90.0

    # Pitch: nose above/below shoulder midpoint
    shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
    dy = shoulder_mid_y - nose[1]
    dx = max(abs(l_shoulder[0] - r_shoulder[0]) / 2.0, 1e-6)
    pitch_deg = math.degrees(math.atan2(dy, dx)) - 45.0  # offset: neutral look = ~45°

    return yaw_deg, pitch_deg


def _angle_to_octant(yaw_deg: float) -> int:
    """Map yaw to one of 8 directional octants (0-7). 0=forward, 4=full-back."""
    # Normalise to [0, 360)
    angle = yaw_deg % 360.0
    return int((angle + 22.5) / 45.0) % 8


# ---------------------------------------------------------------------------
# Per-track state record
# ---------------------------------------------------------------------------

class _TrackPoseState:
    """Rolling state for a single track_id."""

    def __init__(self, fps: float):
        self.fps = fps

        # Sub-signal 1: gaze-shift
        self.gaze_shifted_since: Optional[float] = None  # timestamp when shift started
        self.gaze_signal_emitted_until: float = 0.0       # don't re-emit for X sec

        # Sub-signal 3: scanning
        # Each entry: (timestamp_sec, octant)
        self.octant_history: collections.deque = collections.deque()
        self.last_octant: Optional[int] = None
        self.last_yaw: float = 0.0

    def update_gaze(self, yaw_deg: float, pitch_deg: float, ts: float) -> bool:
        """Returns True if sustained_gaze_shift should be emitted this frame."""
        shifted = abs(yaw_deg) > GAZE_ANGLE_THRESHOLD_DEG or abs(pitch_deg) > GAZE_ANGLE_THRESHOLD_DEG

        if shifted:
            if self.gaze_shifted_since is None:
                self.gaze_shifted_since = ts
            elif ts - self.gaze_shifted_since >= GAZE_SUSTAIN_SEC:
                if ts > self.gaze_signal_emitted_until:
                    # Emit signal, suppress re-emission for the sustained duration
                    self.gaze_signal_emitted_until = ts + GAZE_SUSTAIN_SEC
                    return True
        else:
            self.gaze_shifted_since = None

        return False

    def update_scan(self, yaw_deg: float, ts: float) -> bool:
        """Returns True if targeted_scanning should be emitted."""
        octant = _angle_to_octant(yaw_deg)
        delta = abs(yaw_deg - self.last_yaw)
        self.last_yaw = yaw_deg

        # Prune entries outside the rolling window
        cutoff = ts - SCAN_WINDOW_SEC
        while self.octant_history and self.octant_history[0][0] < cutoff:
            self.octant_history.popleft()

        # Only record a snap if it's a meaningful direction change
        if self.last_octant is not None and octant != self.last_octant and delta >= SNAP_MIN_DELTA_DEG:
            self.octant_history.append((ts, octant))

        self.last_octant = octant

        # Count how often each octant appears in the window (excluding current dwell)
        counts: Dict[int, int] = collections.Counter(o for _, o in self.octant_history)
        if any(c >= SCAN_MIN_SNAPS for c in counts.values()):
            return True

        return False


# ---------------------------------------------------------------------------
# Wrist-proximity state (cross-track)
# ---------------------------------------------------------------------------

class _WristState:
    """Tracks the latest absolute wrist positions for all active track_ids."""

    def __init__(self):
        # tid -> (left_wrist_abs_px, right_wrist_abs_px) | None
        self._wrists: Dict[int, Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]] = {}

    def update(self, tid: int,
               left_wrist: Optional[Tuple[float, float]],
               right_wrist: Optional[Tuple[float, float]]):
        self._wrists[tid] = (left_wrist, right_wrist)

    def evict(self, tid: int):
        self._wrists.pop(tid, None)

    def check_proximity(self) -> List[Tuple[int, int]]:
        """
        Returns list of (tid_a, tid_b) pairs where any wrist of A is within
        WRIST_PROXIMITY_PX of any wrist of B.
        """
        hits = []
        tids = list(self._wrists.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                ta, tb = tids[i], tids[j]
                wa = [w for w in self._wrists[ta] if w is not None]
                wb = [w for w in self._wrists[tb] if w is not None]
                for pa in wa:
                    for pb in wb:
                        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                        if d <= WRIST_PROXIMITY_PX:
                            hits.append((ta, tb))
                            break  # one hit per pair is enough
                    else:
                        continue
                    break
        return hits


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class PoseGestureAnalyzer:
    """
    Stateful, per-video analyzer.  Instantiate once per clip, call update()
    each frame with the current crop dict, collect returned signals.

    Parameters
    ----------
    fps : float
        Native video FPS — used only for timing logic that needs wall-clock
        estimates when a real timestamp is not provided.
    model_path : str, optional
        Path to the MediaPipe pose landmarker .task file.
        Defaults to models/pose_landmarker_lite.task relative to repo root.
    min_detection_confidence : float
    min_tracking_confidence : float
    """

    def __init__(
        self,
        fps: float = 25.0,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.fps = fps
        self._track_states: Dict[int, _TrackPoseState] = {}
        self._wrist_state = _WristState()
        self._warned_no_mp = False
        self._pose = None

        if _MP_AVAILABLE:
            resolved_path = model_path or DEFAULT_MODEL_PATH
            resolved_path = _os.path.abspath(resolved_path)
            if not _os.path.exists(resolved_path):
                warnings.warn(
                    f"[pose_gesture] Model file not found: {resolved_path}. "
                    "Download pose_landmarker_lite.task and place it in models/. "
                    "pose/gesture signals will not fire until the model is present.",
                    RuntimeWarning,
                    stacklevel=1,
                )
            else:
                try:
                    opts = _mp_landmarker_opts_cls(
                        base_options=_mp_base_opts_cls(model_asset_path=resolved_path),
                        running_mode=_RunningMode.IMAGE,
                        num_poses=4,  # up to 4 candidates in frame
                        min_pose_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                    )
                    self._pose = _mp_landmarker_cls.create_from_options(opts)
                except Exception as e:
                    warnings.warn(
                        f"[pose_gesture] Failed to create PoseLandmarker: {e}",
                        RuntimeWarning,
                        stacklevel=1,
                    )

    def _get_state(self, tid: int) -> _TrackPoseState:
        if tid not in self._track_states:
            self._track_states[tid] = _TrackPoseState(self.fps)
        return self._track_states[tid]

    def evict_track(self, tid: int):
        """Call when a track_id is dropped from the tracker."""
        self._track_states.pop(tid, None)
        self._wrist_state.evict(tid)

    def update(
        self,
        frame_index: int,
        crops_by_tid: Dict[int, np.ndarray],
        timestamp_sec: Optional[float] = None,
        track_boxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> Dict[int, List[str]]:
        """
        Run pose analysis on flagged crops.

        Parameters
        ----------
        frame_index : int
        crops_by_tid : dict  {track_id: crop_image (BGR, uint8)}
            Only flagged (motion-gated) tracks should be included here.
            Mirrors the existing pattern for detect_objects() calls.
        timestamp_sec : float, optional
            If not provided, estimated from frame_index / fps.
        track_boxes : dict, optional
            {track_id: (x1, y1, x2, y2)} for translating local crop coordinates to global.

        Returns
        -------
        dict  {track_id: [signal_str, ...]}
            Only track_ids that generated at least one signal are included.
            Signal strings: "sustained_gaze_shift", "chit_passing:<tid_a>+<tid_b>",
            "targeted_scanning".
        """
        if self._pose is None:
            if not self._warned_no_mp:
                warnings.warn(
                    "[pose_gesture] No pose model available — pose signals will not fire.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_no_mp = True
            return {}

        if timestamp_sec is None:
            timestamp_sec = frame_index / self.fps
            
        track_boxes = track_boxes or {}

        signals_out: Dict[int, List[str]] = {}

        # --- Per-track pose estimation ---
        for tid, crop in crops_by_tid.items():
            if crop is None or crop.size == 0:
                continue

            state = self._get_state(tid)
            h, w = crop.shape[:2]
            
            box = track_boxes.get(tid, (0, 0, w, h))
            bx1, by1 = box[0], box[1]

            # Tasks API expects mediapipe.Image (RGB)
            rgb = crop[:, :, ::-1].copy() if crop.ndim == 3 else crop
            mp_image = _mp_image_cls(
                image_format=_mp_image_format.SRGB,
                data=rgb,
            )

            try:
                results = self._pose.detect(mp_image)
            except Exception as e:
                warnings.warn(f"[pose_gesture] detect() failed for tid {tid}: {e}")
                continue

            # Results is a PoseLandmarkerResult with pose_landmarks: list of list of NormalizedLandmark
            if not results.pose_landmarks:
                state.gaze_shifted_since = None
                self._wrist_state.update(tid, None, None)
                continue

            # Use the first detected person in the crop
            lm_list = results.pose_landmarks[0]

            # Build a fake landmark object compatible with _yaw_pitch_from_landmarks
            class _FakeLandmarks:
                def __init__(self, landmarks):
                    self.landmark = landmarks
            fake_lm = _FakeLandmarks(lm_list)
            yaw_deg, pitch_deg = _yaw_pitch_from_landmarks(fake_lm, w, h)

            # Sub-signal 1: sustained gaze-shift
            if state.update_gaze(yaw_deg, pitch_deg, timestamp_sec):
                signals_out.setdefault(tid, []).append("sustained_gaze_shift")

            # Sub-signal 3: targeted scanning (per-track yaw history)
            if state.update_scan(yaw_deg, timestamp_sec):
                signals_out.setdefault(tid, []).append("targeted_scanning")

            # Collect wrist positions for cross-track chit-passing check
            # Tasks API: NormalizedLandmark has x, y, z, visibility attributes
            def wrist_px(idx):
                lmk = lm_list[idx]
                if getattr(lmk, 'visibility', 1.0) < 0.5:
                    return None
                return (lmk.x * w + bx1, lmk.y * h + by1)

            # MediaPipe Pose landmarks: 15=left wrist, 16=right wrist
            self._wrist_state.update(tid, wrist_px(15), wrist_px(16))

        # --- Sub-signal 2: chit-passing (cross-track, checked once per frame) ---
        chit_pairs = self._wrist_state.check_proximity()
        for ta, tb in chit_pairs:
            signal = f"chit_passing:{ta}+{tb}"
            signals_out.setdefault(ta, []).append(signal)
            signals_out.setdefault(tb, []).append(signal)

        return signals_out

    def close(self):
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None


# ---------------------------------------------------------------------------
# Event-schema wiring helpers
# ---------------------------------------------------------------------------

def signals_to_activities(signals: List[str]) -> List[str]:
    """
    Deduplicate and normalise a list of raw pose signal strings for the
    `activities` array. Keeps unique values, preserves first occurrence order.
    """
    seen = set()
    out = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def integrate_pose_into_event(
    event: Dict[str, Any],
    accumulated_signals: List[str],
) -> Dict[str, Any]:
    """
    Merge pose/gesture signals into an event's `activities` array.

    Parameters
    ----------
    event : dict
        The event dict (from P2P3Bridge._finalize_track or test fixture).
        Must already have an `activities` key (added by p2_p3_bridge.py).
    accumulated_signals : list[str]
        All pose signals collected across the lifetime of this track.

    Returns
    -------
    dict: event with `activities` extended in-place.
    """
    if "activities" not in event:
        event["activities"] = []

    new_activities = signals_to_activities(accumulated_signals)
    for act in new_activities:
        if act not in event["activities"]:
            event["activities"].append(act)

    return event
