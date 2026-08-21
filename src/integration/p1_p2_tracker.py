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
except ImportError:
    from ..motion.motion import MotionEstimator
    from ..motion.roi import get_rois
    from ..motion.exclusion_regions import get_exclusion_regions
    from ..track_det.tracker import track, reset_tracker, get_track_history
    from ..track_det.detector import detect_objects
    from ..track_det.fusion import fuse_track_detections
    from ..track_det.invigilator_filter import is_invigilator_track


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
        if frame is not None and boxes:
            try:
                exam_mode = getattr(self, "exam_mode", "CBT")
                crops = [frame[int(y1):int(y2), int(x1):int(x2)]
                         for (x1, y1, x2, y2) in boxes]
                crops = [c for c in crops if c.size > 0]
                if crops:
                    detections = detect_objects(crops, exam_mode)
            except Exception as e:
                print(f"[pipeline] Exception during detect_objects at frame {frame_index}: {e}")
                detections = []

        # 4. Fusion
        fused_tracks = fuse_track_detections(
            tracks, detections, containment_thresh=self.containment_thresh
        )

        # 5. Invigilator Filter
        for ft in fused_tracks:
            tid = ft["track_id"]
            track_hist = get_track_history(tid)
            ft["invigilator_flag"] = is_invigilator_track(track_hist)

        return {
            "frame_index": frame_index,
            "rois": boxes,
            "roi_boxes": boxes,  # backward compatibility alias
            "tracks": tracks,
            "detections": detections,
            "fused_tracks": fused_tracks,
        }

    def reset(self, max_distance: Optional[float] = None, max_age: Optional[int] = None):
        """
        Reset pipeline state for a new clip/video.
        """
        self.motion_estimator.reset()
        reset_tracker(max_distance=max_distance, max_age=max_age)


def process_video_live(
    video_path: str,
    clip_name: Optional[str] = None,
    min_area: int = 500,
    max_distance: Optional[float] = None,
    max_age: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding per-frame P1->P2->P3->P4 integrated results for a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    pipeline = P1P2TrackerPipeline(
        clip_name=clip_name,
        min_area=min_area,
        max_distance=max_distance,
        max_age=max_age,
    )

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, frame_index)
        yield result
        frame_index += 1

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


