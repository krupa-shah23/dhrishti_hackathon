"""
p1_p2_tracker.py
Live integration bridge connecting P1 Motion/ROI with P2 Tracker, P2 Detection, Fusion, and Invigilator Filtering.

Pipeline contract:
FRAME -> P1 get_motion_mask(frame) -> P1 get_rois(mask) -> P2 track(boxes) -> tracks
                                                              │
                                                              ├────────► detect_objects(frame) -> detections
                                                              │                 │
                                                              └────────► P2 fuse_track_detections -> fused_tracks
                                                                                │
                                                                                ▼
                                                                     P2 invigilator_filter -> enriched fused_tracks
"""

from typing import List, Dict, Any, Tuple, Optional, Generator
import cv2
import numpy as np

try:
    from src.motion.motion import MotionEstimator
    from src.motion.roi import get_rois
    from src.motion.exclusion_regions import get_exclusion_regions
    from src.track_det.tracker import track, reset_tracker, get_track_history
    from src.track_det.detector import detect_objects
    from src.track_det.fusion import fuse_track_detections
    from src.track_det.invigilator_filter import is_invigilator_track
    from src.track_det.reid import PersonStore, extract_embedding
    from src.integration.person_adapter import adapt_new_or_updated_persons_for_video
    from src.integration.p2_p3_bridge import P2P3Bridge
    from src.integration.event_adapter import adapt_bridge_events_to_schema
except ImportError:
    from ..motion.motion import MotionEstimator
    from ..motion.roi import get_rois
    from ..motion.exclusion_regions import get_exclusion_regions
    from ..track_det.tracker import track, reset_tracker, get_track_history
    from ..track_det.detector import detect_objects
    from ..track_det.fusion import fuse_track_detections
    from ..track_det.invigilator_filter import is_invigilator_track
    from ..track_det.reid import PersonStore, extract_embedding
    from .person_adapter import adapt_new_or_updated_persons_for_video
    from .p2_p3_bridge import P2P3Bridge
    from .event_adapter import adapt_bridge_events_to_schema


class P1P2TrackerPipeline:
    """
    Live frame-by-frame pipeline connecting P1 motion & ROI extraction,
    P2 tracker, P2 object detector, P2 fusion, and P2 invigilator filter.
    """

    def __init__(
        self,
        clip_name: Optional[str] = None,
        exclusion_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        min_area: int = 500,
        history: int = 500,
        var_threshold: int = 25,
        learning_rate: float = 0.0008,
        use_stabilization: bool = False,
        max_distance: Optional[float] = None,
        max_age: Optional[int] = None,
        containment_thresh: float = 0.5,
    ):
        self.clip_name = clip_name
        if exclusion_regions is not None:
            self.exclusion_regions = exclusion_regions
        elif clip_name:
            self.exclusion_regions = get_exclusion_regions(clip_name)
        else:
            self.exclusion_regions = []

        self.min_area = min_area
        self.containment_thresh = containment_thresh
        self.motion_estimator = MotionEstimator(
            history=history,
            var_threshold=var_threshold,
            learning_rate=learning_rate,
            use_stabilization=use_stabilization,
        )
        reset_tracker(max_distance=max_distance, max_age=max_age)
        self.person_store = PersonStore(path="data/persons_store.json")
        self.track_person_map: Dict[int, str] = {}

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        frame_index: int,
        rois_override: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single video frame sequentially:
        1. Extract P1 ROIs (or use rois_override if provided)
        2. Update P2 tracker with ROI boxes
        3. Detect objects in raw frame via P2 detector
        4. Fuse tracks and detections via P2 fusion
        5. Filter invigilator tracks via P2 invigilator filter
        """
        mask = None
        if rois_override is not None:
            boxes = list(rois_override)
        elif frame is not None:
            mask = self.motion_estimator.get_motion_mask(frame)
            boxes = get_rois(
                mask,
                min_area=self.min_area,
                exclusion_regions=self.exclusion_regions,
            )
        else:
            boxes = []

        # 2. Track (MUST be called on every frame including empty boxes)
        tracks = track(boxes)

        # 3. Object Detection (safely isolated)
        detections = []
        if frame is not None:
            try:
                exam_mode = getattr(self, "exam_mode", "CBT")
                crops = [frame[int(y1):int(y2), int(x1):int(x2)] for (x1, y1, x2, y2) in boxes] if boxes else [frame]
                crops = [c for c in crops if c.size > 0]
                if crops:
                    res = detect_objects(crops, exam_mode)
                    if res:
                        for i, item in enumerate(res):
                            if isinstance(item, (tuple, list)) and len(item) == 3:
                                detections.append(item)
                            elif isinstance(item, dict):
                                b = boxes[i] if i < len(boxes) else (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
                                detections.append((b, item.get("class"), item.get("confidence")))
            except Exception as e:
                print(f"[pipeline] Exception during detect_objects at frame {frame_index}: {e}")
                detections = []

        # 4. Fusion
        fused_tracks = fuse_track_detections(
            tracks, detections, containment_thresh=self.containment_thresh
        )

        # 5. Invigilator Filter & Re-ID
        for ft in fused_tracks:
            tid = ft["track_id"]
            track_hist = get_track_history(tid)
            ft["invigilator_flag"] = is_invigilator_track(track_hist)

            if tid not in self.track_person_map and frame is not None:
                x1, y1, x2, y2 = ft["box"]
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    try:
                        emb = extract_embedding(crop)
                        pid = self.person_store.match_or_create(
                            emb, video_id=self.clip_name or "unknown"
                        )
                        self.track_person_map[tid] = pid
                    except Exception:
                        pass
            if tid in self.track_person_map:
                ft["person_id"] = self.track_person_map[tid]

        return {
            "frame_index": frame_index,
            "rois": boxes,
            "roi_boxes": boxes,  # backward compatibility alias
            "tracks": tracks,
            "detections": detections,
            "fused_tracks": fused_tracks,
            "motion_mask": mask,  # P1's binary mask this frame, None if not computed (e.g. rois_override path)
        }

    def reset(self, max_distance: Optional[float] = None, max_age: Optional[int] = None):
        """
        Reset pipeline state for a new clip/video.
        """
        self.motion_estimator.reset()
        reset_tracker(max_distance=max_distance, max_age=max_age)
        self.track_person_map.clear()

    def save_persons(self, path: Optional[str] = None):
        if self.person_store:
            try:
                self.person_store.save(path)
            except Exception as e:
                print(f"[pipeline] Warning: could not save persons store: {e}")


def process_video_live(
    video_path: str,
    clip_name: Optional[str] = None,
    min_area: int = 500,
    max_distance: Optional[float] = None,
    max_age: Optional[int] = None,
    on_persons_ready=None,   # optional callback: list[schemas.Person] -> None
    on_events_ready=None,    # optional callback: list[schemas.Event] -> None, called incrementally
) -> Generator[Dict[str, Any], None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # sane fallback — some containers report 0 for CAP_PROP_FPS

    pipeline = P1P2TrackerPipeline(
        clip_name=clip_name,
        min_area=min_area,
        max_distance=max_distance,
        max_age=max_age,
    )
    bridge = P2P3Bridge(fps=fps)
    video_id = clip_name or "unknown"

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = (float(total_frames) / fps) if (total_frames and total_frames > 0 and fps > 0) else None

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, frame_index)
        yield result

        # Feed this frame's fused tracks into the event-finalization bridge.
        # P2P3Bridge auto-finalizes any track that's gone stale (missing >
        # missing_threshold frames) — drain whatever finalized THIS frame
        # and post it now, not at video-end, so a crash loses at most the
        # in-flight (not-yet-stale) tracks.
        bridge.process_fused_tracks(
            result["fused_tracks"], frame_index,
            frame=frame, motion_mask=result.get("motion_mask"),
        )
        newly_finalized = bridge.get_completed_events()
        if newly_finalized and on_events_ready is not None:
            # seat_id: not carried by P2P3Bridge's event dict at all today —
            # genuinely blocked on P1's seat-grid mapping, per the status doc.
            # Passing None here is honest, not a placeholder bug.
            adapted = adapt_bridge_events_to_schema(
                newly_finalized,
                video_id=video_id,
                video_duration=video_duration,
            )
            on_events_ready(adapted)

        frame_index += 1

    # Flush any tracks still active when the video ends (e.g. someone still
    # in-frame at the last frame) — these never went "stale" so they'd
    # otherwise be silently dropped.
    bridge.flush()
    tail_events = bridge.get_completed_events()
    if tail_events and on_events_ready is not None:
        adapted = adapt_bridge_events_to_schema(
            tail_events,
            video_id=video_id,
            video_duration=video_duration,
        )
        on_events_ready(adapted)

    pipeline.save_persons()

    persons_for_backend = adapt_new_or_updated_persons_for_video(
        person_store=pipeline.person_store,
        video_id=video_id,
        track_person_map=pipeline.track_person_map,
    )
    if on_persons_ready is not None and persons_for_backend:
        on_persons_ready(persons_for_backend)

    cap.release()


def draw_tracks_overlay(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    roi_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    color_roi: Tuple[int, int, int] = (0, 255, 0),
    color_track: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Debug visualization helper: draws P1 ROI boxes (green) and P2 track boxes with track_id (red) on a frame copy.
    If fused track information (class, invigilator_flag) is present, displays it in text overlay.
    """
    out = frame.copy()
    if roi_boxes:
        for (x1, y1, x2, y2) in roi_boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color_roi, 1)

    for tr in tracks:
        x1, y1, x2, y2 = tr["box"]
        tid = tr["track_id"]
        cls = tr.get("class")
        conf = tr.get("confidence")
        is_inv = tr.get("invigilator_flag", False)

        label_parts = [f"ID:{tid}"]
        if cls is not None:
            conf_str = f"{conf:.2f}" if conf is not None else ""
            label_parts.append(f"{cls} {conf_str}".strip())
        if is_inv:
            label_parts.append("INVIGILATOR")

        label = " | ".join(label_parts)
        color = (255, 0, 0) if is_inv else color_track

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out,
            label,
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return out