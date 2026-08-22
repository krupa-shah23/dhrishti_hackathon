"""
verify_hysteresis.py
Checks whether real N-on/M-off frame-level hysteresis exists in the pipeline,
and measures fragmentation effect on clip 08 (Camera12, 1.5 min).

Method:
1. Run full pipeline on clip 08 using the same approach as verify_dahisar1.
2. Apply current segment_events() (motion+duration gate only — no frame-level hysteresis).
3. Report per-event durations, short-event count, and fragmentation runs.
4. Confirm: is frame-level N-on/M-off hysteresis present? (Diagnosis pass.)
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.integration.event_clustering import segment_events, enrich_event_with_motion_fields
from src.integration.p2_p3_bridge import P2P3Bridge
from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity as _mi
from src.motion.roi import get_rois
from src.motion.exclusion_regions import get_exclusion_regions
from src.track_det.tracker import track, reset_tracker, get_track_history
from src.track_det.fusion import fuse_track_detections
from src.track_det.invigilator_filter import is_invigilator_track
from collections import defaultdict
import cv2

VIDEO     = 'data/drishti/08_seat12_copying.mkv'
CLIP_NAME = '08_seat12_copying.mkv'
CAMERA_ID = 'Camera12'

cap = cv2.VideoCapture(VIDEO)
fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nClip: {CLIP_NAME}, FPS={fps:.1f}, total={total} frames ({total/fps:.1f}s)")

motion_est       = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
exclusion_regions = get_exclusion_regions(CLIP_NAME)
reset_tracker()

# Patch _finalize_track to preserve per-track motion_intensities list
original_finalize = P2P3Bridge._finalize_track
def _patched_finalize(self, track_id):
    td = self.active_tracks[track_id]
    intensities = list(td.get('motion_intensities', []))
    original_finalize(self, track_id)
    self.completed_events[-1]['motion_intensities'] = intensities
P2P3Bridge._finalize_track = _patched_finalize

bridge = P2P3Bridge(missing_threshold=int(fps * 1.5), fps=fps)

print("[Running pipeline on clip 08...]")
for fi in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    mag_map, mask = motion_est.get_motion_mask(frame)
    fused = fuse_motion_signal(mag_map, mask)
    mi    = _mi(fused)
    tagged_rois = get_rois(fused, min_area=500, exclusion_regions=exclusion_regions, camera_id=CAMERA_ID)
    roi_seat_map = {r['bbox']: r.get('seat_id', 'unknown') for r in tagged_rois if isinstance(r, dict)}
    bare_boxes   = [r['bbox'] if isinstance(r, dict) else r for r in tagged_rois]
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
                union = (fx2-fx1)*(fy2-fy1)+(bx2-bx1)*(by2-by1)-inter
                iou   = inter/union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_seat = iou, sid
        ft['seat_id'] = best_seat
    bridge.process_fused_tracks(fused_tracks, fi, motion_intensity=mi)

bridge.flush()
cap.release()
P2P3Bridge._finalize_track = original_finalize

raw_events = bridge.get_completed_events()
print(f"\nTotal raw events from P2P3Bridge: {len(raw_events)}")

# --- Enrich ---
enriched = []
for ev in raw_events:
    intensities = ev.get('motion_intensities', [0.0])
    avg_mi  = float(np.mean(intensities)) if intensities else 0.0
    peak_mi = float(max(intensities))     if intensities else 0.0
    ms = {'avg_motion_intensity': avg_mi, 'peak_intensity': peak_mi, 'mog2_foreground_ratio': 1.0}
    enriched.append(enrich_event_with_motion_fields(ev, ms))

# --- Current gate (motion+duration only, no frame-level hysteresis) ---
confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
print(f"After segment_events (threshold=0.02, min_dur=1.0s): {len(confirmed)} confirmed")

# --- Fragmentation analysis ---
durations = [(ev['event_id'], ev['end_time'] - ev['start_time'],
              ev.get('seat_ids', []), ev['start_time'], ev['end_time'])
             for ev in confirmed]

short = [(eid, dur, seats, t0, t1) for eid, dur, seats, t0, t1 in durations if dur < 2.0]
print(f"\n=== FRAGMENTATION ANALYSIS ===")
print(f"Events <2s (potential fragments): {len(short)} / {len(confirmed)}")
if short:
    print("Short events (all):")
    for eid, dur, seats, t0, t1 in short:
        print(f"  {eid}: {t0:.2f}s-{t1:.2f}s ({dur:.2f}s)  seats={seats}")

# Duration histogram
buckets = defaultdict(int)
for _, dur, _, _, _ in durations:
    buckets[int(dur // 1)] += 1
print("\nDuration histogram:")
for b in sorted(buckets):
    print(f"  {b}-{b+1}s : {'#'*buckets[b]} ({buckets[b]})")

# Fragmentation runs: >=3 consecutive short events at same seat within 5s gaps
ev_sorted = sorted(confirmed, key=lambda e: e['start_time'])
runs = []
i = 0
while i < len(ev_sorted):
    ev  = ev_sorted[i]
    dur = ev['end_time'] - ev['start_time']
    if dur < 2.0 and ev.get('seat_ids'):
        run = [ev]
        j   = i + 1
        while j < len(ev_sorted):
            nxt  = ev_sorted[j]
            ndur = nxt['end_time'] - nxt['start_time']
            gap  = nxt['start_time'] - ev_sorted[j-1]['end_time']
            if ndur < 2.0 and gap < 5.0 and set(nxt.get('seat_ids',[])) & set(ev.get('seat_ids',[])):
                run.append(nxt)
                j += 1
            else:
                break
        if len(run) >= 3:
            runs.append(run)
        i = j
    else:
        i += 1

print(f"\nFragmentation runs (>=3 consecutive short events, same seat, <=5s gaps): {len(runs)}")
for run in runs:
    print(f"  Run of {len(run)} at seats={run[0].get('seat_ids')} "
          f"{run[0]['start_time']:.1f}s-{run[-1]['end_time']:.1f}s:")
    for ev in run:
        d = ev['end_time'] - ev['start_time']
        print(f"    {ev['event_id']}: {ev['start_time']:.2f}-{ev['end_time']:.2f}s ({d:.2f}s)")

print("\nDone.")
