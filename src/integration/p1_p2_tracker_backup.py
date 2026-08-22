"""
p1_p2_tracker.py
Live integration bridge connecting P1 Motion/ROI with P2 Tracker, P2 Detection, Fusion, and Invigilator Filtering.

Pipeline contract:
FRAME -> P1 get_motion_mask(frame) -> P1 get_rois(mask) -> P2 track(boxes) -> tracks
                                                              │
                                                              ├────────► detect_objects(frame)       -> detections   (Stage B)
                                                              ├────────► pose_gesture_analyzer.update() -> signals  (Stage C, §3.6)
                                                              │                 │
                                                              └────────► P2 fuse_track_detections -> fused_tracks
                                                                                │
                                                                                ▼
                                                                     P2 invigilator_filter -> enriched fused_tracks
"""

from typing import List, Dict, Any, Tuple, Optional, Generator
import cv2
import numpy as np
import collections
from pathlib import Path

try:
    from src.motion.motion import MotionEstimator
    from src.motion.roi import get_rois
    from src.motion.exclusion_regions import get_exclusion_regions
    from src.motion.pose_gesture import PoseGestureAnalyzer, integrate_pose_into_event
    from src.track_det.tracker import track, reset_tracker, get_track_history
    from src.track_det.detector import detect_objects
    from src.track_det.fusion import fuse_track_detections
    from src.track_det.invigilator_filter import is_invigilator_track
except ImportError:
    from ..motion.motion import MotionEstimator
    from ..motion.roi import get_rois as _get_rois
    def get_rois(*args, **kwargs):
        res = _get_rois(*args, **kwargs)
        if kwargs.get('return_cleaned'):
            return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
        return [b['bbox'] if isinstance(b, dict) else b for b in res]
    from ..motion.exclusion_regions import get_exclusion_regions
    from ..motion.pose_gesture import PoseGestureAnalyzer, integrate_pose_into_event
    from ..track_det.tracker import track, reset_tracker, get_track_history
    from ..track_det.detector import detect_objects
    from ..track_det.fusion import fuse_track_detections
    from ..track_det.invigilator_filter import is_invigilator_track


# Number of raw (un-skipped) frame crops buffered per active track for
# detect_objects(). Decoupled from effective_fps by design (Krupa/P2 decision):
# occlusion coverage must not collapse to N=1 on long-duration clips that
# get a low sample rate. 7 raw frames ≈ 0.3s at 22fps (07_seat_exchange.mkv),
# enough to catch momentary occlusion lift. Memory: ~70 frames × ~700KB =
# ~50MB peak across 10 active tracks — acceptable; P4 to flag if stress tests
# show pressure.
# P4 note: this proceeds on P2's authority over the occlusion-handling
# contract. Loop P4 in if memory pressure appears during Day 5 stress tests.
N_RAW_WINDOW = 7


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
        self.exam_mode = "CBT"

        if clip_name:
            manifest_path = Path("data/drishti/manifest.csv")
            if manifest_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(manifest_path)
                    row = df[df["filename"] == clip_name]
                    if not row.empty:
                        self.exam_mode = row.iloc[0].get("exam_mode", "CBT")
                except Exception as e:
                    print(f"[pipeline] Failed to load manifest metadata for {clip_name}: {e}")

            if exclusion_regions is not None:
                self.exclusion_regions = exclusion_regions
            else:
                self.exclusion_regions = get_exclusion_regions(clip_name)
        else:
            self.exclusion_regions = exclusion_regions if exclusion_regions is not None else []

        # Window size is a fixed raw-frame buffer, decoupled from effective_fps.
        # See N_RAW_WINDOW constant above for justification.
        self.window_size = N_RAW_WINDOW
        self.track_crop_buffer = collections.defaultdict(lambda: collections.deque(maxlen=self.window_size))
        print(f"[pipeline] Initialized with exam_mode={self.exam_mode}, "
              f"window_size={self.window_size} raw frames (decoupled from fps)")

        self.min_area = min_area
        self.containment_thresh = containment_thresh
        self.motion_estimator = MotionEstimator(
            history=history,
            var_threshold=var_threshold,
            learning_rate=learning_rate,
            use_stabilization=use_stabilization,
        )
        reset_tracker(max_distance=max_distance, max_age=max_age)

        # Stage C: Pose/Gesture analyzer (§3.6). Initialized once per clip.
        # PoseGestureAnalyzer is stateful — do NOT re-create per frame.
        self.pose_analyzer = PoseGestureAnalyzer(fps=25.0)  # 25fps default; real fps unknown at init
        # Accumulate all pose signals seen per track_id over its lifetime.
        # List of signal strings; merged into activities[] at track finalization.
        self.pose_signal_buffer: Dict[int, List[str]] = collections.defaultdict(list)

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
            mag_map, mask = self.motion_estimator.get_motion_mask(frame)
            from src.motion.motion import fuse_motion_signal, motion_intensity as _motion_intensity
            fused_mask = fuse_motion_signal(mag_map, mask)
            frame_motion_intensity = _motion_intensity(fused_mask)
            boxes = get_rois(
                fused_mask,
                min_area=self.min_area,
                exclusion_regions=self.exclusion_regions,
            )
        else:
            boxes = []
            frame_motion_intensity = 0.0

        # get_rois() returns {seat_id, bbox} dicts since the seat-tagging refactor.
        # track() expects bare (x1,y1,x2,y2) tuples — extract bbox here.
        boxes = [b["bbox"] if isinstance(b, dict) else b for b in boxes]

        # 2. Track (MUST be called on every frame including empty boxes)
        tracks = track(boxes)

        # 3. Object Detection (Windowed + Crop-Slicing)
        detections = []
        if frame is not None:
            active_tids = set()
            h, w = frame.shape[:2]
            
            # Extract current crop for each active track
            for tr in tracks:
                tid = tr["track_id"]
                active_tids.add(tid)
                x1, y1, x2, y2 = map(int, tr["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    self.track_crop_buffer[tid].append(crop)
            
            # Clean up stale tracks from buffer, evict from pose analyzer
            for tid in list(self.track_crop_buffer.keys()):
                if tid not in active_tids:
                    del self.track_crop_buffer[tid]
                    self.pose_analyzer.evict_track(tid)

            # Stage C: Pose/Gesture analysis on flagged crops (same gate as Stage B)
            # Pass only the LATEST crop per track (single-frame pose is sufficient;
            # temporal state is maintained inside PoseGestureAnalyzer)
            latest_crops = {}
            track_boxes = {}
            for tr in tracks:
                tid = tr["track_id"]
                track_boxes[tid] = tr["box"]
                crop_list = list(self.track_crop_buffer[tid])
                if crop_list:
                    latest_crops[tid] = crop_list[-1]

            pose_signals_this_frame = self.pose_analyzer.update(
                frame_index=frame_index,
                crops_by_tid=latest_crops,
                timestamp_sec=frame_index / 25.0,
                track_boxes=track_boxes
            )
            for tid, sigs in pose_signals_this_frame.items():
                self.pose_signal_buffer[tid].extend(sigs)
            
            # Call detector per active track using its accumulated crops
            for tid in active_tids:
                crop_list = list(self.track_crop_buffer[tid])
                if len(crop_list) > 0:
                    try:
                        # print(f"[pipeline] Calling detect_objects for tid {tid} with {len(crop_list)} crops")
                        track_dets = detect_objects(crop_list, self.exam_mode)
                        
                        # Translate detection boxes (relative to crop) back to absolute frame coordinates
                        current_box = next((tr["box"] for tr in tracks if tr["track_id"] == tid), None)
                        if current_box and track_dets:
                            bx1, by1, _, _ = map(int, current_box)
                            bx1, by1 = max(0, bx1), max(0, by1)
                            for (bbox, cls_name, conf) in track_dets:
                                rx1, ry1, rx2, ry2 = bbox
                                abs_box = (rx1 + bx1, ry1 + by1, rx2 + bx1, ry2 + by1)
                                detections.append((abs_box, cls_name, conf))
                    except Exception as e:
                        print(f"[pipeline] Exception during detect_objects for tid {tid} at frame {frame_index}: {e}")

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
            # pose_signals: {tid: [signal_str, ...]} for signals fired THIS frame
            # Callers (e.g. P2P3Bridge) should merge these into active track's
            # activities buffer; see integrate_pose_into_event().
            "pose_signals": dict(pose_signals_this_frame) if frame is not None else {},
            # motion_intensity: per-frame foreground-pixel fraction (0-1); pass to
            # P2P3Bridge.process_fused_tracks(motion_intensity=...) for severity scoring.
            "motion_intensity": frame_motion_intensity,
        }

    def reset(self, max_distance: Optional[float] = None, max_age: Optional[int] = None):
        """
        Reset pipeline state for a new clip/video.
        """
        self.motion_estimator.reset()
        self.track_crop_buffer.clear()
        self.pose_signal_buffer.clear()
        self.pose_analyzer.close()
        self.pose_analyzer = PoseGestureAnalyzer(fps=25.0)
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


