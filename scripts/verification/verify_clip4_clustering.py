"""
Verification script: Item 1.3 — spatial+time adjacency clustering test on 04_candidate_talking.mkv.

Bypasses P1P2TrackerPipeline (which calls MediaPipe pose on every frame) and runs
directly against the motion/tracking/fusion/bridge stack only.

Camera12 seats (6 total):
  seat_66: (0, 120, 150, 480)    seat_65: (160, 120, 340, 280)
  seat_64: (330, 130, 470, 380)  seat_63: (470, 160, 640, 400)
  seat_61: (80, 260, 310, 480)   seat_60: (280, 340, 480, 480)

Ground truth:
  Pair 1 (3-12s):    seat_66 <-> seat_61, possible seat_65 involvement
  Pair 2 (72-87s, 97-102s, 140-143s): seat_64 <-> seat_65, should link into one incident
"""

import cv2
import numpy as np
from pathlib import Path

from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity as _motion_intensity
from src.motion.roi import get_rois
from src.motion.exclusion_regions import get_exclusion_regions
from src.motion.grid_config import GRID_CONFIGS, get_grid_config
from src.track_det.tracker import track, reset_tracker, get_track_history
from src.track_det.fusion import fuse_track_detections
from src.track_det.invigilator_filter import is_invigilator_track
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import (
    enrich_event_with_motion_fields,
    segment_events,
    cluster_incidents,
    link_related_events,
)

VIDEO = "data/drishti/04_candidate_talking.mkv"
CLIP_NAME = "04_candidate_talking.mkv"
CAMERA_ID = "Camera12"

if not Path(VIDEO).exists():
    print(f"[ERROR] File not found: {VIDEO}")
    exit(1)

SEATS = GRID_CONFIGS[CAMERA_ID]["seats"]
print(f"\nCamera12 seats ({len(SEATS)} total): {list(SEATS.keys())}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Clip: {CLIP_NAME}, FPS={fps:.2f}, frames={total_frames}, res={frame_w}x{frame_h}")

# Initialize components directly (no MediaPipe/pose)
motion_est = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
exclusion_regions = get_exclusion_regions(CLIP_NAME)
reset_tracker()
# Monkey-patch P2P3Bridge to retain motion_intensities so we can use them in the test
original_finalize = P2P3Bridge._finalize_track
def _patched_finalize(self, track_id):
    track_data = self.active_tracks[track_id]
    intensities = list(track_data.get("motion_intensities", []))
    original_finalize(self, track_id)
    self.completed_events[-1]["motion_intensities"] = intensities
P2P3Bridge._finalize_track = _patched_finalize

bridge = P2P3Bridge(missing_threshold=30, fps=fps)

print("[Running pipeline (no pose — motion/track/fuse/bridge only)...]")
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # P1: motion mask -> fused mask -> ROIs
    mag_map, mask = motion_est.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mask)
    mi = _motion_intensity(fused)

    # get_rois with camera_id for seat tagging
    tagged_boxes = get_rois(
        fused,
        min_area=500,
        exclusion_regions=exclusion_regions,
        camera_id=CAMERA_ID,
    )
    bare_boxes = [b["bbox"] if isinstance(b, dict) else b for b in tagged_boxes]
    # Build bbox->seat_id map for this frame
    roi_seat_map = {b["bbox"]: b.get("seat_id", "unknown") for b in tagged_boxes if isinstance(b, dict)}

    # P2: track, fuse, invigilator filter
    tracks = track(bare_boxes)
    fused_tracks = fuse_track_detections(tracks, [], containment_thresh=0.5)
    for ft in fused_tracks:
        tid = ft["track_id"]
        ft["invigilator_flag"] = is_invigilator_track(get_track_history(tid))
        # Attach seat_id via IOU matching to tagged ROI boxes
        fx1, fy1, fx2, fy2 = ft["box"]
        best_seat = "unknown"
        best_iou = 0.0
        for bbox, seat_id in roi_seat_map.items():
            bx1, by1, bx2, by2 = bbox
            ix1, iy1 = max(fx1, bx1), max(fy1, by1)
            ix2, iy2 = min(fx2, bx2), min(fy2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (fx2-fx1)*(fy2-fy1) + (bx2-bx1)*(by2-by1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_seat = seat_id
        ft["seat_id"] = best_seat

    # Bridge
    bridge.process_fused_tracks(fused_tracks, frame_index, motion_intensity=mi)

    frame_index += 1
    if frame_index % 100 == 0:
        print(f"  ... frame {frame_index}/{total_frames} ({frame_index/fps:.1f}s)")

bridge.flush()
cap.release()

all_events = bridge.get_completed_events()
print(f"\nTotal raw events from P2P3Bridge: {len(all_events)}")

# Use seat_ids now stored directly in each event by P2P3Bridge
def get_event_seats(ev):
    return sorted(ev.get("seat_ids", []))

# Print all raw events
print("\n=== ALL RAW EVENTS (before filtering) ===")
print(f"{'event_id':18} {'tid':5} {'start_s':8} {'end_s':8} {'dur_s':7} {'avg_mi':7} {'seats'}")
print("-" * 100)
for ev in all_events:
    seats = get_event_seats(ev)
    start_s = ev.get("start_frame", 0) / fps
    end_s = ev.get("end_frame", 0) / fps
    dur_s = end_s - start_s
    intensities = ev.get("motion_intensities", [0.0])
    avg_mi = float(np.mean(intensities)) if intensities else 0.0
    print(f"{ev['event_id']:18} {ev['track_id']:5} {start_s:8.2f} {end_s:8.2f} {dur_s:7.2f} {avg_mi:7.4f} {seats}")

# Enrich
enriched = []
for ev in all_events:
    intensities = ev.get("motion_intensities", [0.0])
    avg_mi = float(np.mean(intensities)) if intensities else 0.0
    peak_mi = float(max(intensities)) if intensities else 0.0
    motion_stats = {
        "avg_motion_intensity": avg_mi,
        "peak_intensity": peak_mi,
        "mog2_foreground_ratio": 1.0 if intensities else 0.0,
    }
    enriched.append(enrich_event_with_motion_fields(ev, motion_stats))

confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
print(f"\nAfter segment_events (threshold=0.02, min_dur=1.0s): {len(confirmed)} confirmed")

clusters = cluster_incidents(confirmed, time_gap_sec=5.0)
print(f"After cluster_incidents (time_gap=5.0s): {len(clusters)} clusters")

linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=60.0)

# Print final linked events
print("\n=== FINAL LINKED EVENTS (post-cluster + link) ===")
print(f"{'event_id':18} {'tid':5} {'start_s':8} {'end_s':8} {'dur_s':7} {'related_id':18} {'seats'}")
print("-" * 110)
for ev in sorted(linked, key=lambda e: e.get("start_time", 0)):
    seats = get_event_seats(ev)
    start_s = ev.get("start_time", 0.0)
    end_s = ev.get("end_time", 0.0)
    dur_s = end_s - start_s
    related = ev.get("related_event_id", "—")
    print(f"{ev['event_id']:18} {ev['track_id']:5} {start_s:8.2f} {end_s:8.2f} {dur_s:7.2f} {str(related):18} {seats}")

# ---- Ground truth checks ----
print("\n=== GROUND TRUTH CHECKS ===")

# Pair 1: seat_66 <-> seat_61, window 3-12s
pair1 = [ev for ev in linked
         if any(s in get_event_seats(ev) for s in ["seat_66", "seat_61"])
         and ev.get("start_time", 0) <= 15
         and ev.get("end_time", 0) >= 2]
seat65_in_pair1 = any("seat_65" in get_event_seats(ev) for ev in pair1)
print(f"\nPair 1 (3-12s, seat_66/seat_61): {len(pair1)} events found")
for ev in pair1:
    print(f"  {ev['event_id']}: seats={get_event_seats(ev)}, {ev['start_time']:.2f}s-{ev['end_time']:.2f}s, related={ev.get('related_event_id','—')}")
print(f"  seat_65 in pair 1 cluster? {'YES (ambiguity present)' if seat65_in_pair1 else 'NO'}")

# Pair 2: seat_64/seat_65, windows 72-87s, 97-102s, 140-143s
pair2 = [ev for ev in linked
         if any(s in get_event_seats(ev) for s in ["seat_64", "seat_65"])
         and ev.get("start_time", 0) >= 60]
print(f"\nPair 2 (seat_64/seat_65, windows 72-87s, 97-102s, 140-143s): {len(pair2)} events found")
pair2_event_ids = {ev["event_id"] for ev in pair2}
for ev in sorted(pair2, key=lambda e: e.get("start_time", 0)):
    print(f"  {ev['event_id']}: seats={get_event_seats(ev)}, {ev['start_time']:.2f}s-{ev['end_time']:.2f}s, related={ev.get('related_event_id','—')}")

# Link check: do all pair2 events have related_event_ids pointing to other pair2 events?
pair2_links = [(ev["event_id"], ev.get("related_event_id")) for ev in pair2 if ev.get("related_event_id")]
all_linked = all(rid in pair2_event_ids for _, rid in pair2_links)
print(f"  Three-window cross-linkage: {'CONFIRMED — all events link within pair2' if pair2_links and all_linked else 'NOT fully linked — see related_event_ids above'}")

# False merge check
print(f"\nFalse-merge check (seats not expected in either pair):")
check_seats = {"seat_63", "seat_60"}
for seat in sorted(check_seats):
    false_events = [ev for ev in linked if seat in get_event_seats(ev)]
    if false_events:
        print(f"  [WARN] {seat} appears in {len(false_events)} event(s):")
        for ev in false_events:
            print(f"    {ev['event_id']}: {ev['start_time']:.2f}s-{ev['end_time']:.2f}s, related={ev.get('related_event_id','—')}")
    else:
        print(f"  [OK]   {seat}: not in any linked event")

print("\nDone.")
