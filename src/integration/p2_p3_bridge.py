from typing import List, Dict, Any, Optional
import numpy as np

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
        frame: Optional[np.ndarray] = None,
        motion_mask: Optional[np.ndarray] = None,
    ):
        """
        frame / motion_mask are optional and additive (existing callers that
        don't pass them, e.g. the unit tests, are unaffected — thumbnail/
        heatmap capture below just no-ops without them). When given:
          - frame: current raw BGR frame, used to crop a representative
            thumbnail source per track (best-confidence detection frame if
            any, else most-recently-seen frame).
          - motion_mask: current frame's P1 binary motion mask, accumulated
            per-track over its lifetime as a stopgap "activity heatmap"
            source (see thumbnail_heatmap_writer.py for the real caveat).
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
                    "best_crop": None,
                    "best_crop_confidence": -1.0,
                    "last_crop": None,
                    "heatmap_accum": None,
                }

            track_data = self.active_tracks[track_id]
            track_data["positions"].append(position)
            track_data["boxes"].append(p3_box)
            track_data["frame_indices"].append(frame_index)
            track_data["end_frame"] = frame_index
            track_data["last_seen_frame"] = frame_index

            if is_invigilator:
                track_data["is_invigilator"] = True

            if cls is not None:
                track_data["object_detected"] = True
                track_data["metadata"].append({"frame": frame_index, "class": cls, "confidence": conf})

            # Thumbnail-source capture: crop this track's own box out of the
            # current frame. "last_crop" is the fallback (works for events
            # with no object detection, e.g. talking/seat-exchange); "best_crop"
            # tracks whichever frame had the highest detection confidence,
            # since that's the more literal "flagged moment" when one exists.
            if frame is not None:
                cy1, cy2 = max(0, int(y1)), max(0, int(y2))
                cx1, cx2 = max(0, int(x1)), max(0, int(x2))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    track_data["last_crop"] = crop
                    if conf is not None and conf > track_data["best_crop_confidence"]:
                        track_data["best_crop"] = crop
                        track_data["best_crop_confidence"] = conf

            # Heatmap-source capture: accumulate this frame's motion mask
            # (whole-frame, not just this track's box) over the track's
            # lifetime, so a caller can render "where activity concentrated
            # while this event was open".
            if motion_mask is not None:
                if track_data["heatmap_accum"] is None:
                    track_data["heatmap_accum"] = np.zeros(motion_mask.shape, dtype=np.float32)
                track_data["heatmap_accum"] += (motion_mask > 0).astype(np.float32)

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

        best_detection = None
        if track_data["metadata"]:
            best_detection = max(track_data["metadata"], key=lambda m: m["confidence"] or 0.0)

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
            "is_invigilator": track_data["is_invigilator"],
        }

        if self.audio_path:
            event["audio_path"] = self.audio_path

        if track_data["metadata"]:
            event["metadata"] = track_data["metadata"]

        # Internal-only keys (leading underscore, not part of the P3/schema
        # contract) -- event_adapter.py reads these to actually write the
        # thumbnail/heatmap files, then they're discarded (Event is built
        # from explicit kwargs there, not **event, so these never leak into
        # the POSTed payload).
        thumbnail_frame = track_data["best_crop"] if track_data["best_crop"] is not None else track_data["last_crop"]
        if thumbnail_frame is not None:
            event["_thumbnail_frame"] = thumbnail_frame
        if track_data["heatmap_accum"] is not None:
            event["_heatmap_accum"] = track_data["heatmap_accum"]

        self.completed_events.append(event)

    def get_completed_events(self) -> List[Dict[str, Any]]:
        events = list(self.completed_events)
        self.completed_events.clear()
        return events
