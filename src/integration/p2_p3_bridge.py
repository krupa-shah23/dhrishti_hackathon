from typing import List, Dict, Any, Optional

class P2P3Bridge:
    def __init__(self, missing_threshold: int = 30, fps: float = 30.0, audio_path: Optional[str] = None):
        self.missing_threshold = missing_threshold
        self.fps = fps
        self.audio_path = audio_path
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.completed_events: List[Dict[str, Any]] = []
        self.current_frame = 0

    def process_fused_tracks(self, fused_tracks: List[Dict[str, Any]], frame_index: int):
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
                    "metadata": []
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
