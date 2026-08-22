"""
Verification script: DAHISAR1 spatial+time adjacency clustering.
Clip: 07_seat_exchange.mkv (8h long — only process first 30 minutes, ~39600 frames at 22fps).
Ground truth: seat exchange event — two candidates swap seats, should produce
cross-linked events between adjacent seat pairs.
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.integration.event_clustering import cluster_incidents, segment_events, enrich_event_with_motion_fields, link_related_events
from src.integration.p2_p3_bridge import P2P3Bridge
from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity as _mi
from src.motion.roi import get_rois
from src.motion.exclusion_regions import get_exclusion_regions, CAMERA_CONFIGS
from src.motion.grid_config import GRID_CONFIGS
from src.track_det.tracker import track, reset_tracker, get_track_history
from src.track_det.fusion import fuse_track_detections
from src.track_det.invigilator_filter import is_invigilator_track
import cv2

VIDEO = 'data/drishti/07_seat_exchange.mkv'
CLIP_NAME = '07_seat_exchange.mkv'
CAMERA_ID = 'DAHISAR1'
# 30 minutes = 1800s * 22fps = 39600 frames
MAX_FRAMES = 39600

SEATS = GRID_CONFIGS[CAMERA_ID]['seats']
print(f"\nDAHISAR1 seats ({len(SEATS)} total): {list(SEATS.keys())}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
run_frames = min(total_frames, MAX_FRAMES)
print(f"Clip: {CLIP_NAME}, FPS={fps:.2f}, total={total_frames}, processing first {run_frames} frames ({run_frames/fps/60:.1f} min)")

motion_est = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
exclusion_regions = get_exclusion_regions(CLIP_NAME)
cam_key = (CAMERA_ID, (frame_w, frame_h))
if cam_key in CAMERA_CONFIGS:
    exclusion_regions.extend(CAMERA_CONFIGS[cam_key])
reset_tracker()

original_finalize = P2P3Bridge._finalize_track
def _patched_finalize(self, track_id):
    td = self.active_tracks[track_id]
    intensities = list(td.get('motion_intensities', []))
    original_finalize(self, track_id)
    self.completed_events[-1]['motion_intensities'] = intensities
P2P3Bridge._finalize_track = _patched_finalize

bridge = P2P3Bridge(missing_threshold=30, fps=fps)

print("[Running pipeline...]")
for fi in range(run_frames):
    ret, frame = cap.read()
    if not ret:
        break
    mag_map, mask = motion_est.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mask)
    mi = _mi(fused)
    tagged_boxes = get_rois(fused, min_area=500, exclusion_regions=exclusion_regions, camera_id=CAMERA_ID)
    bare_boxes = [b['bbox'] if isinstance(b, dict) else b for b in tagged_boxes]
    roi_seat_map = {b['bbox']: b.get('seat_id', 'unknown') for b in tagged_boxes if isinstance(b, dict)}
    tracks = track(bare_boxes)
    fused_tracks = fuse_track_detections(tracks, [], containment_thresh=0.5)
    for ft in fused_tracks:
        tid = ft['track_id']
        ft['invigilator_flag'] = is_invigilator_track(get_track_history(tid))
        fx1, fy1, fx2, fy2 = ft['box']
        best_seat, best_iou = 'unknown', 0.0
        for bbox, sid in roi_seat_map.items():
            bx1, by1, bx2, by2 = bbox
            ix1, iy1 = max(fx1, bx1), max(fy1, by1)
            ix2, iy2 = min(fx2, bx2), min(fy2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = (fx2-fx1)*(fy2-fy1) + (bx2-bx1)*(by2-by1) - inter
                iou = inter/union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_seat = iou, sid
        ft['seat_id'] = best_seat
    bridge.process_fused_tracks(fused_tracks, fi, motion_intensity=mi)
    if fi % 2000 == 0 and fi > 0:
        print(f"  ... frame {fi}/{run_frames} ({fi/fps/60:.1f}min)")

bridge.flush()
cap.release()
P2P3Bridge._finalize_track = original_finalize

import json
all_events = bridge.get_completed_events()
print(f"\nTotal raw events from P2P3Bridge: {len(all_events)}")
with open('dahisar1_events.json', 'w') as f:
    json.dump(all_events, f, indent=2)

def get_event_seats(ev):
    return sorted(ev.get('seat_ids', []))

enriched = []
for ev in all_events:
    intensities = ev.get('motion_intensities', [0.0])
    avg_mi = float(np.mean(intensities)) if intensities else 0.0
    peak_mi = float(max(intensities)) if intensities else 0.0
    ms = {'avg_motion_intensity': avg_mi, 'peak_intensity': peak_mi, 'mog2_foreground_ratio': 1.0}
    enriched.append(enrich_event_with_motion_fields(ev, ms))

confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
print(f"After segment_events (threshold=0.02, min_dur=1.0s): {len(confirmed)} confirmed")

clusters = cluster_incidents(confirmed, time_gap_sec=5.0, camera_id=CAMERA_ID)
print(f"After cluster_incidents (time_gap=5.0s): {len(clusters)} clusters")

linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=60.0, camera_id=CAMERA_ID)

# Seat attribution summary
seats_seen = {}
for ev in linked:
    for s in ev.get('seat_ids', []):
        seats_seen[s] = seats_seen.get(s, 0) + 1

print(f"\n=== SEAT ATTRIBUTION SUMMARY ===")
print(f"Unique seats seen in events: {sorted(seats_seen.keys())}")
for s, cnt in sorted(seats_seen.items()):
    print(f"  {s}: {cnt} events")

# Cross-linked events by seat pair
print(f"\n=== CROSS-LINKED EVENTS (events with related_event_id) ===")
has_related = [(ev, ev.get('related_event_id')) for ev in linked if ev.get('related_event_id')]
print(f"Total events with related_event_id: {len(has_related)}")

# Group by seat pairs
pair_counts = {}
for ev, rel_id in has_related:
    seats = tuple(sorted(ev.get('seat_ids', ['unknown'])))
    pair_counts[seats] = pair_counts.get(seats, 0) + 1

print("Events with related_event_id by seat:")
for seat_tuple, cnt in sorted(pair_counts.items()):
    print(f"  {seat_tuple}: {cnt} events")

# Isolation check: any seat_7 appearing in related links?
seat7_linked = [ev for ev in linked if 'seat_7' in ev.get('seat_ids', []) and ev.get('related_event_id')]
print(f"\nIsolation check: seat_7 (invigilator) in related links: {len(seat7_linked)} events")

# Show events around any exchange activity (look for events with adjacent seat linkage)
print(f"\n=== ADJACENT-SEAT LINKED EVENT PAIRS (sample) ===")
adj_map = {
    'seat_1': {'seat_2'}, 'seat_2': {'seat_1','seat_3'},
    'seat_3': {'seat_2','seat_4'}, 'seat_4': {'seat_3','seat_5'},
    'seat_5': {'seat_4','seat_6'}, 'seat_6': {'seat_5'},
}
shown = 0
ev_by_id = {ev['event_id']: ev for ev in linked}
for ev in sorted(linked, key=lambda e: e.get('start_time', 0)):
    rel_id = ev.get('related_event_id')
    if not rel_id:
        continue
    rel_ev = ev_by_id.get(rel_id)
    if not rel_ev:
        continue
    seats_a = set(ev.get('seat_ids', []))
    seats_b = set(rel_ev.get('seat_ids', []))
    # Check if they are genuinely adjacent (different seats, adjacency map says so)
    is_adj = False
    for sa in seats_a:
        if seats_b & adj_map.get(sa, set()):
            is_adj = True
    if is_adj and seats_a != seats_b and shown < 20:
        s_a = ev.get('start_time', 0)
        e_a = ev.get('end_time', 0)
        print(f"  {ev['event_id']}(seats={sorted(seats_a)},{s_a:.1f}s-{e_a:.1f}s) <-> {rel_id}(seats={sorted(seats_b)})")
        shown += 1

print(f"\nDone.")
