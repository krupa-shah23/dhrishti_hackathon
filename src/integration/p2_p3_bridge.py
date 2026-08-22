from typing import List, Dict, Any, Optional

try:
    from src.motion.severity_scoring import compute_severity, score_to_risk_label
except ImportError:
    from ..motion.severity_scoring import compute_severity, score_to_risk_label

class P2P3Bridge:
    def __init__(self, missing_threshold: int = 30, fps: float = 30.0, audio_path: Optional[str] = None):
        self.missing_threshold = missing_threshold
        self.fps = fps
        self.audio_path = audio_path
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.completed_events: List[Dict[str, Any]] = []
        self.current_frame = 0

    def process_fused_tracks(
        self,
        fused_tracks: List[Dict[str, Any]],
        frame_index: int,
        pose_signals: Optional[Dict[int, List[str]]] = None,
        motion_intensity: Optional[float] = None,
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
                    "object_detected": False,
                    "is_invigilator": False,
                    "metadata": [],
                    "pose_activities": [],  # accumulated pose/gesture signals for this track
                    "motion_intensities": [],  # per-frame motion intensity (fraction of pixels)
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

            # Accumulate per-frame motion intensity for severity scoring (§3.7)
            if motion_intensity is not None:
                track_data["motion_intensities"].append(float(motion_intensity))

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
            "is_invigilator": track_data["is_invigilator"],
            # activities: populated from pose/gesture signals accumulated during the track's lifetime
            "activities": list(track_data.get("pose_activities", [])),
            # seat_ids: dominant seat regions for this track (seats seen in >25% of frames).
            # Uses per-seat frame-count logic to prevent transient box-growth bleed.
            "seat_ids": dominant_seats,
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

        self.completed_events.append(event)

    def get_completed_events(self) -> List[Dict[str, Any]]:
        events = list(self.completed_events)
        self.completed_events.clear()
        return events
