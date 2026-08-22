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

§4 coordinate contract (Phase 1 lock): process_frame() performs NO implicit
resize. Every ROI/track/detection/event coordinate this module produces is in
the SAME resolution as the `frame` array the caller passed in -- which, for
every real caller in this repo (process_video_live(), the verification
scripts under scripts/verification/, ablation_study.py, benchmark_stages.py),
is the ORIGINAL/NATIVE resolution read straight from cv2.VideoCapture, since
none of them route frames through frame_stream.py's 480p downsize step first.
Native resolution varies by camera/clip (e.g. 640x480 for
04_candidate_talking.mkv vs 1280x720 for most other clips) -- there is no
single fixed "processing resolution" to assume. Seat-grid attribution
(grid_config.get_grid_config()) already scales correctly to whatever
w_img/h_img the mask/frame actually has, so this holds regardless of which
camera/clip is being processed. Any future consumer of these coordinates
(bbox_overlay, a frontend overlay, etc.) must read the resolution from the
same per-video metadata already tracked in manifest.csv/get_video_metadata(),
not assume 480p or any other fixed size. See
test_p1_p2_live.py::test_process_frame_uses_native_input_resolution_no_implicit_resize.
"""

from typing import List, Dict, Any, Tuple, Optional, Generator
import cv2
import numpy as np
import collections
from pathlib import Path

try:
    from src.motion.motion import MotionEstimator
    from src.motion.roi import get_rois
    from src.motion.exclusion_regions import get_exclusion_regions, get_exclusion_mask
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
    from ..motion.exclusion_regions import get_exclusion_regions, get_exclusion_mask
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
        fps: float = 25.0,
    ):
        self.clip_name = clip_name
        self.exam_mode = "CBT"
        # Real source fps for this clip, used for PoseGestureAnalyzer's internal
        # timestamp_sec math (frame_index / fps). Defaults to 25.0 -- the value
        # every existing caller relied on implicitly before this parameter
        # existed -- so behavior is unchanged unless a caller opts in with the
        # clip's real fps (see get_video_metadata()/cap.get(cv2.CAP_PROP_FPS)).
        # Non-25fps clips (e.g. 04_candidate_talking.mkv @8fps,
        # 07_seat_exchange.mkv @22fps) get wrong pose/gesture timing until a
        # caller passes the real value here -- tracked as a Phase 2 follow-up,
        # not silently changed now (would alter already-validated §3.6 output).
        self.fps = fps

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

        # Tracks which (camera_id, width, height) the motion_estimator's
        # camera_mask currently reflects, so process_frame() only recomputes
        # the mask when the camera or frame resolution actually changes.
        self._camera_mask_key: Optional[Tuple[str, int, int]] = None

        # Stage C: Pose/Gesture analyzer (§3.6). Initialized once per clip.
        # PoseGestureAnalyzer is stateful — do NOT re-create per frame.
        self.pose_analyzer = PoseGestureAnalyzer(fps=self.fps)
        # Accumulate all pose signals seen per track_id over its lifetime.
        # List of signal strings; merged into activities[] at track finalization.
        self.pose_signal_buffer: Dict[int, List[str]] = collections.defaultdict(list)

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        frame_index: int,
        rois_override: Optional[List[Tuple[float, float, float, float]]] = None,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single video frame sequentially:
        1. Extract P1 ROIs (or use rois_override if provided)
        2. Update P2 tracker with ROI boxes
        3. Detect objects in raw frame via P2 detector
        4. Fuse tracks and detections via P2 fusion
        5. Filter invigilator tracks via P2 invigilator filter
        """
        # seat_id_map: maps ROI bbox tuple -> seat_id, built from get_rois() dicts.
        # Carried forward to fused_tracks so P2P3Bridge can store correct seat attribution.
        roi_seat_map: Dict[Tuple, str] = {}

        if rois_override is not None:
            boxes = list(rois_override)
            # No motion detection was performed in override mode -- there is no
            # real intensity value to report. Matches the frame-is-None branch
            # below, which makes the same choice for the same reason.
            frame_motion_intensity = 0.0
            frame_mog2_ratio = 0.0
        elif frame is not None:
            if camera_id is not None:
                h, w = frame.shape[:2]
                mask_key = (camera_id, w, h)
                if self._camera_mask_key != mask_key:
                    self.motion_estimator.camera_mask = get_exclusion_mask(camera_id, w, h)
                    self._camera_mask_key = mask_key
            mag_map, mask = self.motion_estimator.get_motion_mask(frame)
            from src.motion.motion import fuse_motion_signal, motion_intensity as _motion_intensity
            fused_mask = fuse_motion_signal(mag_map, mask)
            frame_motion_intensity = _motion_intensity(fused_mask)
            # §6 motion-metric contract: `mask` here is MOG2's own cleaned binary
            # output, before the OR-fusion with frame-diff above. This is the
            # only point in the pipeline where the genuine MOG2-only signal is
            # still distinguishable from the fused one, so this is the correct
            # layer to compute a real mog2_foreground_ratio (fraction of frame
            # pixels MOG2 itself flagged as foreground) rather than faking one
            # from the fused mask downstream.
            frame_mog2_ratio = _motion_intensity(mask)
            tagged_rois = get_rois(
                fused_mask,
                min_area=self.min_area,
                exclusion_regions=self.exclusion_regions,
                camera_id=camera_id,  # None is fine — get_rois defaults to 'unknown' for all seats
            )
            # Build bbox→seat_id map BEFORE stripping to bare tuples for track()
            for roi in tagged_rois:
                if isinstance(roi, dict):
                    roi_seat_map[roi["bbox"]] = roi.get("seat_id", "unknown")
            boxes = [roi["bbox"] if isinstance(roi, dict) else roi for roi in tagged_rois]
        else:
            boxes = []
            frame_motion_intensity = 0.0
            frame_mog2_ratio = 0.0

        # 2. Track (MUST be called on every frame including empty boxes)
        tracks = track(boxes)

        # 3. Object Detection (Windowed + Crop-Slicing)
        detections = []
<<<<<<< HEAD
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
                timestamp_sec=frame_index / self.fps,
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
=======
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
>>>>>>> origin/p2a-p2b-merge

        # 4. Fusion
        fused_tracks = fuse_track_detections(
            tracks, detections, containment_thresh=self.containment_thresh
        )

        # 5. Invigilator Filter + seat_id passthrough
        # Match each fused track back to the ROI box with highest IOU to recover seat_id.
        for ft in fused_tracks:
            tid = ft["track_id"]
            track_hist = get_track_history(tid)
            ft["invigilator_flag"] = is_invigilator_track(track_hist)

            # Attach seat_id: find the ROI box that best matches this track's current box
            if roi_seat_map:
                fx1, fy1, fx2, fy2 = ft["box"]
                best_seat = "unknown"
                best_iou = 0.0
                for bbox, seat_id in roi_seat_map.items():
                    bx1, by1, bx2, by2 = bbox
                    ix1, iy1 = max(fx1, bx1), max(fy1, by1)
                    ix2, iy2 = min(fx2, bx2), min(fy2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = ((fx2-fx1)*(fy2-fy1) + (bx2-bx1)*(by2-by1) - inter)
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_seat = seat_id
                ft["seat_id"] = best_seat
            else:
                ft["seat_id"] = "unknown"

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
            # mog2_foreground_ratio: per-frame MOG2-only foreground-pixel fraction
            # (0-1), measured before OR-fusion with frame-diff. Pass to
            # P2P3Bridge.process_fused_tracks(mog2_foreground_ratio=...) so the
            # §6 event field of the same name can be computed from a genuine
            # measurement instead of a placeholder constant.
            "mog2_foreground_ratio": frame_mog2_ratio,
        }

    def reset(self, max_distance: Optional[float] = None, max_age: Optional[int] = None):
        """
        Reset pipeline state for a new clip/video.
        """
        self.motion_estimator.reset()
        self.track_crop_buffer.clear()
        self.pose_signal_buffer.clear()
        self.pose_analyzer.close()
        self.pose_analyzer = PoseGestureAnalyzer(fps=self.fps)
        reset_tracker(max_distance=max_distance, max_age=max_age)


def process_video_live(
    video_path: str,
    clip_name: Optional[str] = None,
    min_area: int = 500,
    max_distance: Optional[float] = None,
    max_age: Optional[int] = None,
    camera_id: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding per-frame P1->P2->P3->P4 integrated results for a video file.
    camera_id: passed through to process_frame() for per-camera exclusion masking
    (exclusion_regions.get_exclusion_mask) and seat-grid attribution
    (grid_config.get_grid_config). None is fine — behaves as before (no masking,
    seat_id='unknown').
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
        result = pipeline.process_frame(frame, frame_index, camera_id=camera_id)
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


