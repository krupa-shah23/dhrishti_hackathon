"""
test_roi.py
Owner: P1 - Motion & ROI

Runs motion + ROI together on each clip, saves:
- overlay sample frames (boxes drawn on original frames) for visual check
- a CSV per clip: frame_index, x1, y1, x2, y2 — one row per detected ROI,
  ready to hand off to P2's tracker.
"""

import os
import csv
import glob
import cv2
import numpy as np

try:
    from .motion import MotionEstimator
    from .roi import get_rois as _get_rois, draw_rois
    def get_rois(*args, **kwargs):
        res = _get_rois(*args, **kwargs)
        if kwargs.get('return_cleaned'):
            return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
        return [b['bbox'] if isinstance(b, dict) else b for b in res]
    from .exclusion_regions import get_exclusion_regions
except ImportError:
    from motion import MotionEstimator
    from roi import get_rois as _get_rois, draw_rois
    def get_rois(*args, **kwargs):
        res = _get_rois(*args, **kwargs)
        if kwargs.get('return_cleaned'):
            return [b['bbox'] if isinstance(b, dict) else b for b in res[0]], res[1]
        return [b['bbox'] if isinstance(b, dict) else b for b in res]
    from exclusion_regions import get_exclusion_regions

try:
    from .roi import get_grid_rois
    from .grid_config import get_grid_config
except ImportError:
    from roi import get_grid_rois
    from grid_config import get_grid_config


import unittest


class TestGetGridRois(unittest.TestCase):
    """
    Plan 1.5: get_grid_rois() unit tests. Synthetic masks against Camera12's
    real grid (native 640x480, no scaling) -- seat_65/seat_64 are the
    already-verified adjacent pair from the closed clip-4 clustering work
    (grid_config.py's own adjacency map), reused here rather than inventing
    new synthetic seat IDs.
    """

    def setUp(self):
        self.grid = get_grid_config("Camera12")  # native 640x480, matches mask shape below
        self.seats = self.grid["seats"]

    def test_two_adjacent_cells_produce_two_separate_boxes(self):
        # One CONTIGUOUS blob straddling seat_65 (160,120,340,280) and
        # seat_64 (330,130,470,380) -- exactly the shape a contour-based
        # get_rois() would merge into a single box (losing one seat
        # entirely). get_grid_rois() must still emit two independent boxes.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[150:300, 200:400] = 255  # crosses the seat_65/seat_64 boundary at x=330-340

        boxes = get_grid_rois(mask, self.grid)
        seat_ids = sorted(b["seat_id"] for b in boxes)

        self.assertEqual(seat_ids, ["seat_64", "seat_65"],
                          "expected exactly one box per straddled seat, not a merged single box")
        for b in boxes:
            self.assertEqual(b["bbox"], self.seats[b["seat_id"]],
                              "bbox must be the seat's own rectangle, not a contour-derived box")

    def test_scattered_subthreshold_noise_is_rejected(self):
        # seat_66: (0, 120, 150, 480), area 150*360=54000. Scatter isolated
        # 3x3 dots (non-touching, 3px gaps) totaling well over 5% of the
        # seat's area, so the intensity gate passes -- but the largest
        # single contour is only 9px, far under MIN_CONTOUR_AREA=500, so
        # shape-validity must still reject it.
        mask = np.zeros((480, 640), dtype=np.uint8)
        x1, y1, x2, y2 = self.seats["seat_66"]
        for yy in range(y1 + 10, y1 + 190, 6):
            for xx in range(x1 + 10, x2 - 10, 6):
                mask[yy:yy + 3, xx:xx + 3] = 255

        total_on = int(np.count_nonzero(mask[y1:y2, x1:x2]))
        area = (x2 - x1) * (y2 - y1)
        self.assertGreaterEqual(total_on / area, 0.05,
                                 "test setup bug: scattered noise must clear the intensity gate")

        boxes = get_grid_rois(mask, self.grid)
        self.assertNotIn("seat_66", [b["seat_id"] for b in boxes],
                          "scattered sub-MIN_CONTOUR_AREA noise should be rejected by shape validity")

    def test_single_contiguous_blob_fills_cell_boundary_used_not_contour_bbox(self):
        # seat_63: (470, 160, 640, 400). One solid blob covering most of it.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:380, 490:620] = 255  # smaller than the full cell -- proves bbox isn't the contour's own box

        boxes = get_grid_rois(mask, self.grid)
        matching = [b for b in boxes if b["seat_id"] == "seat_63"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["bbox"], self.seats["seat_63"],
                          "bbox must equal the full cell rectangle, not the tighter contour bounding box")

    def test_zero_motion_cell_is_skipped_at_intensity_gate(self):
        # seat_60: (280, 340, 480, 480) left entirely zero; motion placed
        # elsewhere (seat_63) so the mask isn't trivially all-empty.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:380, 490:620] = 255

        boxes = get_grid_rois(mask, self.grid)
        self.assertNotIn("seat_60", [b["seat_id"] for b in boxes])

    def test_crop_bbox_clips_at_cell_edge(self):
        # seat_63: (470, 160, 640, 400), area 40800 -- needs >=2040px to
        # clear the 5% intensity gate, so the corner blob is 50x50=2500px
        # (not just big enough to test clipping, big enough to be detected
        # at all). Blob touches the top-left corner exactly -- a naive -9px
        # margin would push past x1/y1; crop_bbox must clip to the cell's
        # own edge instead.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[160:210, 470:520] = 255  # touches x1=470 and y1=160 exactly

        boxes = get_grid_rois(mask, self.grid)
        matching = [b for b in boxes if b["seat_id"] == "seat_63"]
        self.assertEqual(len(matching), 1)
        cx1, cy1, cx2, cy2 = matching[0]["crop_bbox"]
        self.assertEqual((cx1, cy1), (470, 160), "must clip to the cell's top-left edge, not go negative-relative")
        self.assertEqual((cx2, cy2), (529, 219), "far side gets the full +9px margin since it's nowhere near an edge")

        # Same check on the bottom-right corner, seat_66 (0,120,150,480),
        # area 54000 -- needs >=2700px, so a 55x55=3025px corner blob.
        mask2 = np.zeros((480, 640), dtype=np.uint8)
        mask2[425:480, 95:150] = 255  # touches x2=150 and y2=480 exactly
        boxes2 = get_grid_rois(mask2, self.grid)
        matching2 = [b for b in boxes2 if b["seat_id"] == "seat_66"]
        self.assertEqual(len(matching2), 1)
        cx1b, cy1b, cx2b, cy2b = matching2[0]["crop_bbox"]
        self.assertEqual((cx2b, cy2b), (150, 480), "must clip to the cell's bottom-right edge")
        self.assertEqual((cx1b, cy1b), (86, 416), "near side gets the full +9px margin since it's nowhere near an edge")

    def test_crop_bbox_no_clipping_needed_when_centered(self):
        # seat_66: (0, 120, 150, 480), area 54000 -- 55x55=3025px blob
        # (clears the 5% gate) placed well away from every edge. crop_bbox
        # should be exactly the contour bbox +/- 9px, untouched.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[240:295, 40:95] = 255

        boxes = get_grid_rois(mask, self.grid)
        matching = [b for b in boxes if b["seat_id"] == "seat_66"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["crop_bbox"], (31, 231, 104, 304))

    def test_crop_bbox_degenerate_full_cell_contour_no_crash(self):
        # seat_63: (470, 160, 640, 400) filled entirely -- the contour's own
        # bounding box already equals the full cell on all four sides, so
        # the +9px margin would want to exceed every edge simultaneously.
        # Must degrade gracefully to exactly the cell rect, no crash.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[160:400, 470:640] = 255

        boxes = get_grid_rois(mask, self.grid)
        matching = [b for b in boxes if b["seat_id"] == "seat_63"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["crop_bbox"], (470, 160, 640, 400))
        self.assertEqual(matching[0]["crop_bbox"], matching[0]["bbox"])


CLIP_PATHS = {
    "cubicle": "data/shadow/cubicle/input",
    "backdoor": "data/shadow/backdoor/input",
    "copyMachine": "data/shadow/copyMachine/input",
    "tramCrossroad_1fps": "data/lowFramerate/tramCrossroad_1fps/input",
    "fountain02": "data/dynamicBackground/fountain02/input",
    "badminton": "data/cameraJitter/badminton/input",
    "boulevard": "data/cameraJitter/boulevard/input",
    "sidewalk": "data/cameraJitter/sidewalk/input",
    "traffic": "data/cameraJitter/traffic/input",
    "shanghai_01_001": "data/shanghaitech/testing/frames/01_001",
    "shanghai_01_0015": "data/shanghaitech/testing/frames/01_0015",
}

OEP_BASE = "data/OEP database"
OEP_MIN_AREA = 1500  # webcam close-up framing makes hair/glasses/fabric texture noise
                      # clear MIN_CONTOUR_AREA=500 (tuned for CDNet's smaller, farther
                      # subjects); validated via sweep_min_area against subject1 frame
                      # 14000 (hand-raise gesture) — box survives intact at this value
                      # while most hair/collar fragmentation clears. See sweep results
                      # in outputs/benchmark_logs/subject1_ma*_rois.csv for the full trend.
OEP_SUBJECTS = ["subject1", "subject10", "subject11", "subject12", "subject13",
                "subject14", "subject15", "subject16", "subject17", "subject18",
                "subject19", "subject2", "subject20", "subject21", "subject22",
                "subject23", "subject24", "subject3", "subject4", "subject5",
                "subject6", "subject7", "subject8", "subject9"]

def find_oep_webcam_file(subject_folder):
    """
    OEP video filenames are username-based (e.g. Yousef1.avi = webcam,
    Yousef2.avi = wearcam) — only the webcam file (ends in '1.avi') is used;
    the wearcam is head-mounted and incompatible with a static-camera pipeline.
    """
    candidates = glob.glob(os.path.join(subject_folder, "*1.avi"))
    if not candidates:
        raise FileNotFoundError(f"No webcam (*1.avi) file found in {subject_folder}")
    return candidates[0]

MASK_OUT_DIR = "outputs/masks"
LOG_OUT_DIR = "outputs/benchmark_logs"
SAMPLE_EVERY_N_FRAMES = 20


def load_frames_from_folder(folder_path):
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".bmp")))
    for fname in files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            yield fname, frame


def load_frames_from_video(video_path):
    """
    Frame loader for genuine video files (MSU OEP .avi), parallel to
    load_frames_from_folder() which handles CDNet/ShanghaiTech image sequences.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield f"frame_{i:06d}", frame
        i += 1
    cap.release()

def run_roi_check(clip_name, folder_path, exclusion_regions=None, suffix="",
                   use_stabilization=False, is_video=False, var_threshold=25,
                   min_area=500):
    """for i, (fname, original_frame) in enumerate(load_frames_from_folder(folder_path)):
    Runs ROI detection on a clip and writes overlay samples + a CSV.

    exclusion_regions: list of (x1,y1,x2,y2) tuples passed straight to
        get_rois(); pass [] or None to run without any exclusion mask.
    suffix: appended to output directory name and CSV filename so that
        before/after runs don't overwrite each other (e.g. '_no_mask',
        '_with_mask').
    """
    label = clip_name + suffix
    print(f"\n--- Running ROI check on: {label} ---")
    path_exists = os.path.isfile(folder_path) if is_video else os.path.isdir(folder_path)
    if not path_exists:
        print(f"  [SKIP] {'File' if is_video else 'Folder'} not found: {folder_path}")
        return None

    out_dir = os.path.join(MASK_OUT_DIR, label)
    _probe = load_frames_from_video(folder_path) if is_video else load_frames_from_folder(folder_path)
    for _fname, _frame in _probe:
        print(f"  Frame resolution: {_frame.shape[1]}x{_frame.shape[0]}")
        break
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOG_OUT_DIR, exist_ok=True)

    estimator = MotionEstimator(use_stabilization=use_stabilization, var_threshold=var_threshold)
    csv_path = os.path.join(LOG_OUT_DIR, f"{label}_rois.csv")
    total_boxes = 0
    frame_count = 0
    shifts = []  # (dx, dy) per frame, only meaningful when use_stabilization=True

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "x1", "y1", "x2", "y2"])

        frame_source = load_frames_from_video(folder_path) if is_video else load_frames_from_folder(folder_path)
        for i, (fname, original_frame) in enumerate(frame_source):
            mag_map, mask = estimator.get_motion_mask(original_frame)
            
            from motion import fuse_motion_signal
            fused_mask = fuse_motion_signal(mag_map, mask)
            
            save_sample = (i % SAMPLE_EVERY_N_FRAMES == 0)

            if use_stabilization:
                shifts.append(estimator.last_shift)

            if save_sample:
                boxes, cleaned = get_rois(fused_mask, min_area=min_area, return_cleaned=True,
                                          exclusion_regions=exclusion_regions)
            else:
                boxes = get_rois(fused_mask, min_area=min_area, exclusion_regions=exclusion_regions)
                cleaned = None

            total_boxes += len(boxes)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"    Processed {frame_count} frames...")

            for (x1, y1, x2, y2) in boxes:
                writer.writerow([i, x1, y1, x2, y2])

            if save_sample:
                overlay = draw_rois(original_frame, boxes)
                cv2.imwrite(os.path.join(out_dir, f"roi_sample_{i:05d}.jpg"), overlay)
                cv2.imwrite(os.path.join(out_dir, f"cleaned_mask_{i:05d}.png"), cleaned)

                if use_stabilization:
                    dx, dy = estimator.last_shift
                    h, w = original_frame.shape[:2]
                    translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    stabilized_frame = cv2.warpAffine(
                        original_frame, translation_matrix, (w, h),
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    diff = cv2.absdiff(original_frame, stabilized_frame)
                    cv2.imwrite(os.path.join(out_dir, f"original_{i:05d}.jpg"), original_frame)
                    cv2.imwrite(os.path.join(out_dir, f"stabilized_{i:05d}.jpg"), stabilized_frame)
                    cv2.imwrite(os.path.join(out_dir, f"diff_{i:05d}.jpg"), diff)

    avg = total_boxes / frame_count if frame_count else 0
    print(f"  Frames processed : {frame_count}")
    print(f"  Total ROIs detected : {total_boxes}")
    print(f"  Avg ROIs/frame : {avg:.3f}")
    print(f"  ROI overlay samples : {out_dir}")
    print(f"  ROI CSV (P2 handoff) : {csv_path}")

    result = {"clip": label, "frames": frame_count,
              "total_rois": total_boxes, "avg_rois": avg}

    if use_stabilization and shifts:
        mags = [ (dx**2 + dy**2) ** 0.5 for dx, dy in shifts ]
        avg_shift = sum(mags) / len(mags)
        max_shift = max(mags)
        print(f"  Avg translation (px) : {avg_shift:.2f}")
        print(f"  Max translation (px) : {max_shift:.2f}")
        result["avg_shift"] = avg_shift
        result["max_shift"] = max_shift

    return result


def sweep_var_threshold(clip_name, folder_path, thresholds=(16, 20, 25), is_video=False):
    """
    Runs the same clip at multiple MOG2 varThreshold values, everything else
    fixed, to test whether lowering sensitivity recovers low-contrast misses
    (e.g. light clothing on light pavement) without reintroducing false
    positives. Manually cross-check overlay samples at the frames where a
    miss was previously observed (e.g. 00120/00140/00160/00200 for
    shanghai_01_0015) — avg ROIs/frame alone won't tell you if the specific
    miss was fixed or if new false positives appeared elsewhere.

    is_video: True for OEP-style .avi inputs (see find_oep_webcam_file),
    False for CDNet/ShanghaiTech image-sequence folders.
    """
    print(f"\n{'='*60}")
    print(f"  varThreshold SWEEP: {clip_name}")
    print(f"{'='*60}")

    results = []
    for vt in thresholds:
        r = run_roi_check(clip_name, folder_path,
                          suffix=f"_vt{vt}", var_threshold=vt, is_video=is_video)
        if r:
            results.append((vt, r))

    print(f"\n  --- Sweep summary ---")
    print(f"  {'varThreshold':<15} {'Avg ROIs/frame':>16}")
    print(f"  {'-'*35}")
    for vt, r in results:
        print(f"  {vt:<15} {r['avg_rois']:>16.3f}")
    print(f"\n  Next: manually check overlay samples in outputs/masks/"
          f"{clip_name}_vt<N>/ at the frames where misses were previously "
          f"observed, to confirm the miss is actually fixed, not just that "
          f"the average changed.")

def sweep_min_area(clip_name, folder_path, areas=(500, 1000, 1500, 2000), is_video=False, var_threshold=25):
    """
    Runs the same clip at multiple MIN_CONTOUR_AREA values, var_threshold held
    fixed at whatever value the varThreshold sweep settled on. Tests whether
    raising the area floor clears texture-noise blobs (hair/glasses/fabric on
    a close-framed webcam subject) without eating the real face/torso box.
    """
    print(f"\n{'='*60}")
    print(f"  min_area SWEEP: {clip_name}")
    print(f"{'='*60}")

    results = []
    for ma in areas:
        r = run_roi_check(clip_name, folder_path,
                          suffix=f"_ma{ma}", min_area=ma,
                          var_threshold=var_threshold, is_video=is_video)
        if r:
            results.append((ma, r))

    print(f"\n  --- Sweep summary ---")
    print(f"  {'min_area':<15} {'Avg ROIs/frame':>16}")
    print(f"  {'-'*35}")
    for ma, r in results:
        print(f"  {ma:<15} {r['avg_rois']:>16.3f}")
    print(f"\n  Next: check outputs/masks/{clip_name}_ma<N>/roi_sample_03560.jpg "
          f"specifically — confirm fragments clear without losing the face/torso box.")
    
def compare_exclusion_mask(clip_name, folder_path, exclusion_regions):
    """
    Runs the same clip twice — once without and once with the exclusion mask —
    and prints a side-by-side summary so we can measure impact.
    """
    print(f"\n{'='*60}")
    print(f"  EXCLUSION MASK COMPARISON: {clip_name}")
    print(f"{'='*60}")

    before = run_roi_check(clip_name, folder_path,
                           exclusion_regions=None,
                           suffix="_no_mask")
    after  = run_roi_check(clip_name, folder_path,
                           exclusion_regions=exclusion_regions,
                           suffix="_with_mask")

    if before and after:
        delta_total = before["total_rois"] - after["total_rois"]
        delta_avg   = before["avg_rois"]   - after["avg_rois"]
        pct         = (delta_total / before["total_rois"] * 100
                       if before["total_rois"] else 0)
        print(f"\n  --- Comparison summary ---")
        print(f"  {'Metric':<28} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Total ROIs':<28} {before['total_rois']:>10} {after['total_rois']:>10} {-delta_total:>+10}")
        print(f"  {'Avg ROIs/frame':<28} {before['avg_rois']:>10.3f} {after['avg_rois']:>10.3f} {-delta_avg:>+10.3f}")
        print(f"  {'ROI reduction':<28} {'':<10} {'':<10} {pct:>9.1f}%")
        print(f"\n  Interpretation: negative delta = fewer detections (desired for FP zones).")
        print(f"  Verify overlay images to confirm no true detections were lost.")


def compare_stabilization(clip_name, folder_path):
    """
    Runs the same clip with and without camera-shake compensation and
    prints a side-by-side summary — expected result: avg ROIs/frame drops
    sharply once stabilization is on, if the clip has vibration-driven
    full-frame false positives.
    """
    print(f"\n{'='*60}")
    print(f"  STABILIZATION COMPARISON: {clip_name}")
    print(f"{'='*60}")

    before = run_roi_check(clip_name, folder_path,
                           suffix="_unstabilized", use_stabilization=False)
    after  = run_roi_check(clip_name, folder_path,
                           suffix="_stabilized", use_stabilization=True)

    if before and after:
        delta_avg = before["avg_rois"] - after["avg_rois"]
        pct = (delta_avg / before["avg_rois"] * 100) if before["avg_rois"] else 0
        print(f"\n  --- Comparison summary ---")
        print(f"  {'Metric':<28} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Avg ROIs/frame':<28} {before['avg_rois']:>10.3f} {after['avg_rois']:>10.3f} {-delta_avg:>+10.3f}")
        if "avg_shift" in after:
            print(f"  {'Avg translation (px)':<28} {'':<10} {after['avg_shift']:>10.2f}")
            print(f"  {'Max translation (px)':<28} {'':<10} {after['max_shift']:>10.2f}")
        print(f"\n  ROI reduction: {pct:.1f}%")
        print(f"  Interpretation: large drop = shake-driven full-frame ROIs suppressed.")


<<<<<<< HEAD
def run_ignore_regions_check(clip_name, folder_path, regions):
=======
def check_ignore_regions(clip_name, folder_path, regions):
>>>>>>> origin/p2a-p2b-merge
    """
    Sanity check for the invigilator ignore-region hook: runs one clip with
    a fixed region blanked out and confirms motion boxes never appear inside it.
    """
    print(f"\n{'='*60}")
    print(f"  IGNORE-REGION HOOK TEST: {clip_name}")
    print(f"{'='*60}")

    if not os.path.isdir(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}")
        return

    estimator = MotionEstimator()
    estimator.set_ignore_regions(regions)

    violations = 0
    frame_count = 0
    rx1, ry1, rx2, ry2 = regions[0]


    for i, (fname, frame) in enumerate(load_frames_from_folder(folder_path)):
        mag_map, mask = estimator.get_motion_mask(frame)
        boxes = get_rois(mask)
        frame_count += 1
        for (x1, y1, x2, y2) in boxes:
            # box center inside the ignored region = leak
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                violations += 1

    print(f"  Frames checked : {frame_count}")
    print(f"  Region tested  : {regions[0]}")
    print(f"  Boxes leaking into ignored region : {violations}")
    print(f"  Result: {'PASS' if violations == 0 else 'FAIL'}")


if __name__ == "__main__":
    print("test_roi.py started")


    # --- Day-3 experiment: before/after comparison on copyMachine only -------
    cm_path = CLIP_PATHS["copyMachine"]
    cm_excl = get_exclusion_regions("copyMachine")
    compare_exclusion_mask("copyMachine", cm_path, cm_excl)

    for clip_name in ["badminton", "boulevard", "sidewalk", "traffic"]:
        compare_stabilization(clip_name, CLIP_PATHS[clip_name])

<<<<<<< HEAD
    run_ignore_regions_check("cubicle", CLIP_PATHS["cubicle"], [(0, 0, 150, 480)])
=======
    check_ignore_regions("cubicle", CLIP_PATHS["cubicle"], [(0, 0, 150, 480)])
>>>>>>> origin/p2a-p2b-merge

    sweep_var_threshold("shanghai_01_0015", CLIP_PATHS["shanghai_01_0015"])

    try:
        subj1_video = find_oep_webcam_file(os.path.join(OEP_BASE, "subject1"))
        sweep_var_threshold("subject1", subj1_video, thresholds=(25, 40, 55), is_video=True)
        sweep_min_area("subject1", subj1_video, areas=(500, 1000, 1500, 2000), is_video=True)

        print(f"\n{'='*60}")
        print(f"  OEP PILOT RUN")
        print(f"{'='*60}")
        for subj in ["subject1", "subject10"]:  # subject1 = actor, subject10 = real exam-taker
            subj_folder = os.path.join(OEP_BASE, subj)
            video_path = find_oep_webcam_file(subj_folder)
            run_roi_check(subj, video_path, is_video=True, min_area=OEP_MIN_AREA)
    except FileNotFoundError as e:
        print(f"\n  [SKIP] OEP dataset not found: {e}")


    # --- Standard run for all other clips (no exclusion mask yet) ------------
    print(f"\n{'='*60}")
    print("  STANDARD RUN (all clips, no exclusion mask)")
    print(f"{'='*60}")
    for name, path in CLIP_PATHS.items():
        if name == "copyMachine":
            # Run with the validated mask so the saved CSV is the clean version
            run_roi_check(name, path,
                          exclusion_regions=get_exclusion_regions(name))
        else:
            run_roi_check(name, path)
    def test_roi_seat_id_tagging(self):
        import numpy as np
        from src.motion.roi import _get_rois
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Create a rect inside seat_61 (Camera12 grid)
        mask[300:400, 150:250] = 255
        
        boxes = _get_rois(mask, camera_id='Camera12', min_area=100)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['seat_id'], 'seat_61')
        self.assertTrue('bbox' in boxes[0])
