"""
p1_p2_tracker.py
Live integration bridge connecting P1 Motion/ROI with P2 Tracker.

Pipeline contract:
VIDEO FRAME -> P1 get_motion_mask(frame) -> P1 get_rois(mask) -> P2 track(boxes) -> tracks

Stateful requirements:
- Single MotionEstimator instance per clip (preserves MOG2 background model)
- Single tracker state per clip (track state preserved between frames)
- reset_tracker() called when starting a new clip
"""

from typing import List, Dict, Any, Tuple, Optional, Generator
import cv2
import numpy as np

try:
    from src.motion.motion import MotionEstimator
    from src.motion.roi import get_rois
    from src.motion.exclusion_regions import get_exclusion_regions
    from src.track_det.tracker import track, reset_tracker
except ImportError:
    from ..motion.motion import MotionEstimator
    from ..motion.roi import get_rois
    from ..motion.exclusion_regions import get_exclusion_regions
    from ..track_det.tracker import track, reset_tracker


class P1P2TrackerPipeline:
    """
    Live frame-by-frame pipeline connecting P1 motion & ROI extraction directly to P2 tracker.
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
    ):
        self.clip_name = clip_name
        if exclusion_regions is not None:
            self.exclusion_regions = exclusion_regions
        elif clip_name:
            self.exclusion_regions = get_exclusion_regions(clip_name)
        else:
            self.exclusion_regions = []

        self.min_area = min_area
        self.motion_estimator = MotionEstimator(
            history=history,
            var_threshold=var_threshold,
            learning_rate=learning_rate,
            use_stabilization=use_stabilization,
        )
        reset_tracker(max_distance=max_distance, max_age=max_age)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """
        Processes a single video frame sequentially.
        """
        mask = self.motion_estimator.get_motion_mask(frame)
        boxes = get_rois(
            mask,
            min_area=self.min_area,
            exclusion_regions=self.exclusion_regions,
        )
        tracks = track(boxes)

        return {
            "frame_index": frame_index,
            "roi_boxes": boxes,
            "tracks": tracks,
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
    Generator yielding per-frame P1->P2 tracking results for a video file.
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
    """
    out = frame.copy()
    if roi_boxes:
        for (x1, y1, x2, y2) in roi_boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color_roi, 1)

    for tr in tracks:
        x1, y1, x2, y2 = tr["box"]
        tid = tr["track_id"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color_track, 2)
        cv2.putText(
            out,
            f"Track {tid}",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_track,
            2,
        )
    return out

