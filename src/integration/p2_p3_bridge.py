from typing import List, Dict, Any, Optional, Set

try:
    from src.motion.severity_scoring import compute_severity, score_to_risk_label
    from src.motion.heuristics import flag_seat_vacant_near_invigilator, build_explanation
except ImportError:
    from ..motion.severity_scoring import compute_severity, score_to_risk_label
    from ..motion.heuristics import flag_seat_vacant_near_invigilator, build_explanation

# §5 person-ID contract (Phase 1 lock): the only identity concept this bridge
# (or anything upstream of it -- CentroidTracker in src/track_det/tracker.py)
# produces is `track_id`: an int, transient per-clip, reassigned from 1 on
# every P1P2TrackerPipeline.reset()/new video, and NOT stable across a
# ROI-free gap longer than the tracker's max_age (a departing-and-returning
# subject gets a NEW track_id, per test_p1_p2_live.py's own long-gap case).
# There is no persistent person_id / cross-video / cross-camera identity
# anywhere in this pipeline, and no Re-ID is implemented. Fields named
# person_id/person_ids appear only in Person B's own speculative mock test
# payloads (test_complete_signal.py, test_idempotent_event_id.py) for a
# backend endpoint contract that does not consume this bridge's real output
# -- they are not wired to anything here. Any future backend/API layer that
# needs person_id should treat it as an alias of this event's `track_id`,
# scoped per-video, with the transience above made explicit to callers --
# not a signal of persistent identity.
class P2P3Bridge:
    def __init__(self, missing_threshold: int = 30, fps: float = 30.0, audio_path: Optional[str] = None,
                 camera_id: Optional[str] = None):
        self.missing_threshold = missing_threshold
        self.fps = fps
        self.audio_path = audio_path
        # §3.4 seat_vacant_near_invigilator: which camera's seat-adjacency map to use for
        # Gate 1 (see heuristics.flag_seat_vacant_near_invigilator). None (default) keeps
        # existing callers behaviorally unchanged -- the whole feature is
        # opt-in via this constructor argument.
        self.camera_id = camera_id
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.completed_events: List[Dict[str, Any]] = []
        self.current_frame = 0
        # §3.4 seat_vacant_near_invigilator rolling histories (see process_fused_tracks).
        # Bounded to missing_threshold frames -- the same "how much recent
        # history matters" window already used elsewhere in this class.
        self._invigilator_cell_history: List[Set[str]] = []
        self._seat_activity_history: Dict[str, List[bool]] = {}
        self._last_track_id_per_seat: Dict[str, int] = {}

    def process_fused_tracks(
        self,
        fused_tracks: List[Dict[str, Any]],
        frame_index: int,
        pose_signals: Optional[Dict[int, List[str]]] = None,
        motion_intensity: Optional[float] = None,
        mog2_foreground_ratio: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        fused_tracks : list
            Output of fuse_track_detections().
        frame_index : int
        pose_signals : dict {track_id: [signal_str, ...]}, optional
            Pose/gesture signals from PoseGestureAnalyzer.update() for this frame.
            If provided, signals are accumulated in the track's activities list.
        """
        self.current_frame = frame_index
        seen_track_ids = set()

        for ft in fused_tracks:
            track_id = ft["track_id"]
            seen_track_ids.add(track_id)

            box = ft["box"] # (x1, y1, x2, y2)
            x1, y1, x2, y2 = box

            # P3 compatible box: (x, y, w, h)
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)
            p3_box = (x, y, w, h)

            # Centroid
            cx = float(x1 + x2) / 2.0
            cy = float(y1 + y2) / 2.0
            position = (cx, cy)

            cls = ft.get("class")
            conf = ft.get("confidence")
            is_invigilator = ft.get("invigilator_flag", False)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "event_id": f"event_{track_id}",
                    "track_id": track_id,
                    "positions": [],
                    "boxes": [],
                    "frame_indices": [],
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "last_seen_frame": frame_index,
                    "object_detected": None,
                    "is_invigilator": False,
                    "metadata": [],
                    "pose_activities": [],  # accumulated pose/gesture signals for this track
                    "motion_intensities": [],  # per-frame fused-mask motion intensity (fraction of pixels)
                    "mog2_ratios": [],  # per-frame MOG2-only foreground ratio (§6), pre-fusion
                    # seat_id_counts: per-seat frame-count for dominant-seat logic (Bug 3 fix).
                    # A track that temporarily grows into an adjacent seat's pixel region
                    # should not be attributed to that seat permanently.
                    "seat_id_counts": {},  # seat_id -> int (frame count with that seat as best-match)
                }

            track_data = self.active_tracks[track_id]
            track_data["positions"].append(position)
            track_data["boxes"].append(p3_box)
            track_data["frame_indices"].append(frame_index)
            track_data["end_frame"] = frame_index
            track_data["last_seen_frame"] = frame_index

            # Accumulate per-seat frame counts (dominant-seat logic, Bug 3 fix)
            seat_id = ft.get("seat_id", "unknown")
            if seat_id and seat_id != "unknown":
                counts = track_data["seat_id_counts"]
                counts[seat_id] = counts.get(seat_id, 0) + 1
                # §3.4: remember which track currently owns this seat, so a
                # seat_vacant_near_invigilator tag (fired below, after the seat's
                # track may already have gone quiet this frame) has somewhere to land.
                if not is_invigilator:
                    self._last_track_id_per_seat[seat_id] = track_id

            # Accumulate per-frame motion intensity for severity scoring (§3.7)
            if motion_intensity is not None:
                track_data["motion_intensities"].append(float(motion_intensity))

            # Accumulate per-frame MOG2-only foreground ratio (§6 motion-metric contract)
            if mog2_foreground_ratio is not None:
                track_data["mog2_ratios"].append(float(mog2_foreground_ratio))

            if is_invigilator:
                track_data["is_invigilator"] = True

            if cls is not None:
                track_data["object_detected"] = True
                track_data["metadata"].append({"frame": frame_index, "class": cls, "confidence": conf})

            # Accumulate pose/gesture signals for this track (Stage C, §3.6)
            if pose_signals and track_id in pose_signals:
                for sig in pose_signals[track_id]:
                    if sig not in track_data["pose_activities"]:
                        track_data["pose_activities"].append(sig)

        # --- §3.4 seat_vacant_near_invigilator: update rolling histories, then tag ---
        # Opt-in: only runs when camera_id was supplied (needed for the seat
        # adjacency map Gate 1 checks). Writes into the SAME activities[]
        # array pose/gesture signals already use (see _finalize_track).
        if self.camera_id is not None:
            invigilator_cells_this_frame = {
                ft.get("seat_id") for ft in fused_tracks
                if ft.get("invigilator_flag") and ft.get("seat_id") not in (None, "unknown")
            }
            student_seats_this_frame = {
                ft.get("seat_id") for ft in fused_tracks
                if not ft.get("invigilator_flag") and ft.get("seat_id") not in (None, "unknown")
            }

            self._invigilator_cell_history.append(invigilator_cells_this_frame)
            if len(self._invigilator_cell_history) > self.missing_threshold:
                self._invigilator_cell_history.pop(0)

            # Update every seat we've ever seen a student track occupy, not
            # just this frame's -- a "dip" is precisely the frames where a
            # previously-active seat drops OUT of student_seats_this_frame.
            tracked_seats = set(self._seat_activity_history.keys()) | student_seats_this_frame
            for seat_id in tracked_seats:
                history = self._seat_activity_history.setdefault(seat_id, [])
                history.append(seat_id in student_seats_this_frame)
                if len(history) > self.missing_threshold:
                    history.pop(0)

                if flag_seat_vacant_near_invigilator(history, self._invigilator_cell_history, seat_id, self.camera_id):
                    tid = self._last_track_id_per_seat.get(seat_id)
                    if tid is not None and tid in self.active_tracks:
                        tagged_track = self.active_tracks[tid]
                        if "seat_vacant_near_invigilator" not in tagged_track["pose_activities"]:
                            tagged_track["pose_activities"].append("seat_vacant_near_invigilator")

        # Finalize stale tracks
        stale_ids = []
        for track_id, track_data in self.active_tracks.items():
            if track_id not in seen_track_ids:
                if frame_index - track_data["last_seen_frame"] > self.missing_threshold:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            self._finalize_track(track_id)

    def flush(self):
        for track_id in list(self.active_tracks.keys()):
            self._finalize_track(track_id)

    def _finalize_track(self, track_id: int):
        track_data = self.active_tracks.pop(track_id)
        start_frame = track_data["start_frame"]
        end_frame = track_data["end_frame"]

        # --- Severity Scoring (§3.7) ---
        duration_sec = (end_frame - start_frame) / self.fps if self.fps > 0 else 0.0

        # Mean motion intensity over the track's lifetime; 0.0 if not accumulated.
        intensities = track_data.get("motion_intensities", [])
        mean_motion = float(sum(intensities) / len(intensities)) if intensities else 0.0

        # Repetition: number of object-detection hits accumulated.
        repetition_count = len(track_data.get("metadata", []))

        # Object confidence: max confidence across detection hits (booster, not gate).
        confs = [
            m["confidence"] for m in track_data.get("metadata", [])
            if m.get("confidence") is not None
        ]
        max_obj_confidence = max(confs) if confs else None

        best_detection = None
        if track_data["metadata"]:
            best_detection = max(track_data["metadata"], key=lambda m: m["confidence"] or 0.0)

        severity_score = compute_severity(
            motion_intensity=mean_motion,
            duration=duration_sec,
            repetition_count=repetition_count,
            object_confidence=max_obj_confidence,
        )
        risk_label, color_tag, confidence_int = score_to_risk_label(severity_score)
        # --------------------------------

        # Compute dominant seat attribution from per-seat frame counts.
        # Include: the primary seat (max frames) + any seat seen in >25% of frames.
        # This prevents a temporarily-expanded tracker box from falsely attributing
        # an event to a seat it only touched for a few frames.
        seat_counts = track_data.get("seat_id_counts", {})
        total_seat_frames = sum(seat_counts.values()) if seat_counts else 0
        dominant_seats = []
        if seat_counts:
            max_count = max(seat_counts.values())
            threshold = max(1, total_seat_frames * 0.25)  # >25% of seat-tagged frames
            dominant_seats = sorted(
                [s for s, c in seat_counts.items() if c >= threshold],
                key=lambda s: -seat_counts[s]
            )
            if not dominant_seats:
                # Fallback: no seat individually cleared the threshold (frames spread
                # thinly across several seats) -- force-include the primary seat
                # (max frames) per this function's own stated intent above, so
                # dominant_seats is never empty when at least one frame was
                # seat-attributed at all. Ties at max_count are all included
                # (deterministic; no arbitrary single-winner pick).
                dominant_seats = sorted(
                    [s for s, c in seat_counts.items() if c == max_count],
                    key=lambda s: -seat_counts[s]
                )

        event = {
            "event_id": track_data["event_id"],
            "track_id": track_data["track_id"],
            "positions": track_data["positions"],
            "boxes": track_data["boxes"],
            "frame_indices": track_data["frame_indices"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / self.fps,
            "end_time": end_frame / self.fps,
            "total_frames": end_frame - start_frame + 1,
            "object_detected": track_data["object_detected"],
            "object_confidence": best_detection["confidence"] if best_detection else None,
            # Node's Event.objectDetected is a class-name String (e.g. "phone"),
            # not a bool -- best_detection already carries "class" from
            # detect_objects()'s per-frame metadata, just wasn't surfaced before.
            "object_class": best_detection["class"] if best_detection else None,
            "is_invigilator": track_data["is_invigilator"],
            # activities: populated from pose/gesture signals accumulated during the track's lifetime
            "activities": list(track_data.get("pose_activities", [])),
            # seat_ids: dominant seat regions for this track (seats seen in >25% of frames).
            # Uses per-seat frame-count logic to prevent transient box-growth bleed.
            "seat_ids": dominant_seats,
            # §6 motion-metric contract — raw per-frame series, exposed so
            # enrich_event_with_motion_fields() callers can compute genuine
            # avg_motion_intensity/peak_intensity/mog2_foreground_ratio instead
            # of a hardcoded placeholder. Previously only reachable by
            # monkey-patching _finalize_track (see scripts/verification/*.py).
            "motion_intensities": list(intensities),
            "mog2_ratios": list(track_data.get("mog2_ratios", [])),
            # §9 integration contract — severity fields
            "severity_score": severity_score,
            "risk_label": risk_label,
            "color_tag": color_tag,
            "confidence": confidence_int,
        }

        if self.audio_path:
            event["audio_path"] = self.audio_path

        if track_data["metadata"]:
            event["metadata"] = track_data["metadata"]

        # §9 integration contract — explanation field, same wiring point as
        # the severity fields above (built last since it reads 'metadata',
        # which is only attached to `event` immediately above this line).
        event["explanation"] = build_explanation(event)

        self.completed_events.append(event)

    def get_completed_events(self) -> List[Dict[str, Any]]:
        events = list(self.completed_events)
        self.completed_events.clear()
        return events
