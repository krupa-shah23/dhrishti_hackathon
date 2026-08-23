"""
verify_dominant_seat_threshold_sweep.py
Verifies the max_count fallback fix in P2P3Bridge._finalize_track()
(src/integration/p2_p3_bridge.py, dominant_seats computation).

Runs the real pipeline ONCE per clip (Camera12 clip 4, DAHISAR1 clip 07,
same 30-min window used throughout this session), patching _finalize_track
to additionally capture each track's raw `seat_id_counts` dict onto the
finalized event (same monkey-patch pattern already used elsewhere this
session for motion_intensities).

With real seat_id_counts captured once, the OLD (pre-fix, no fallback) and
NEW (post-fix, with max_count fallback) dominant-seat logic are both
reimplemented standalone here -- byte-identical to the respective code
paths -- and swept over threshold in {0.25, 0.30, 0.35} without needing to
re-run the pipeline per threshold.

Checks:
  1. NEW == OLD at threshold=0.25 (must be a true no-op at the shipped value).
  2. NEW recovers previously-empty (OLD-empty) cases at 0.30/0.35 via the
     max_count fallback, with specific before/after examples.
  3. Ground-truth ratio (Camera12 Pair 1/Pair 2, DAHISAR1 adjacency linkage)
     recomputed at each (threshold, old/new) combination using the actual
     event_clustering pipeline stages, to confirm 0.25 behavior is unaffected
     end-to-end, not just at the dominant_seats field itself.
"""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from src.integration.event_clustering import segment_events, enrich_event_with_motion_fields, cluster_incidents, link_related_events
from src.integration.p2_p3_bridge import P2P3Bridge
from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity as _mi
from src.motion.roi import get_rois
from src.motion.exclusion_regions import get_exclusion_regions, CAMERA_CONFIGS
from src.track_det.tracker import track, reset_tracker, get_track_history
from src.track_det.fusion import fuse_track_detections
from src.track_det.invigilator_filter import is_invigilator_track
import cv2

CLIPS = {
    "clip4": dict(
        video="data/drishti/04_candidate_talking.mkv",
        clip_name="04_candidate_talking.mkv",
        camera_id="Camera12",
        max_frames=None,
        missing_threshold=30,
    ),
    "dahisar1": dict(
        video="data/drishti/07_seat_exchange.mkv",
        clip_name="07_seat_exchange.mkv",
        camera_id="DAHISAR1",
        max_frames=39600,
        missing_threshold=30,
    ),
}


def compute_dominant_seats(seat_counts, threshold_frac, apply_fallback):
    """Standalone mirror of _finalize_track's dominant-seat block."""
    if not seat_counts:
        return []
    total_seat_frames = sum(seat_counts.values())
    max_count = max(seat_counts.values())
    threshold = max(1, total_seat_frames * threshold_frac)
    dominant = sorted([s for s, c in seat_counts.items() if c >= threshold],
                       key=lambda s: -seat_counts[s])
    if not dominant and apply_fallback:
        dominant = sorted([s for s, c in seat_counts.items() if c == max_count],
                           key=lambda s: -seat_counts[s])
    return dominant


def run_pipeline_capture_seat_counts(cfg):
    video = cfg["video"]; camera_id = cfg["camera_id"]; clip_name = cfg["clip_name"]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("[WARN] Invalid FPS, falling back to 22.0")
        fps = 22.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    run_frames = min(total_frames, cfg["max_frames"]) if cfg["max_frames"] else total_frames

    motion_est = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
    exclusion_regions = get_exclusion_regions(clip_name)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_key = (camera_id, (frame_w, frame_h))
    if cam_key in CAMERA_CONFIGS:
        exclusion_regions.extend(CAMERA_CONFIGS[cam_key])
    reset_tracker()

    # Patch _finalize_track to ALSO preserve seat_id_counts + motion_intensities
    # onto the finalized event (production _finalize_track already computes
    # seat_id_counts internally but does not expose it on the event dict).
    original_finalize = P2P3Bridge._finalize_track
    def _patched_finalize(self, track_id):
        td = self.active_tracks[track_id]
        intensities = list(td.get('motion_intensities', []))
        seat_id_counts = dict(td.get('seat_id_counts', {}))
        original_finalize(self, track_id)
        self.completed_events[-1]['motion_intensities'] = intensities
        self.completed_events[-1]['_captured_seat_id_counts'] = seat_id_counts
    P2P3Bridge._finalize_track = _patched_finalize

    bridge = P2P3Bridge(missing_threshold=cfg["missing_threshold"], fps=fps)

    for fi in range(run_frames):
        ret, frame = cap.read()
        if not ret:
            break
        mag_map, mask = motion_est.get_motion_mask(frame)
        fused = fuse_motion_signal(mag_map, mask)
        mi = _mi(fused)
        tagged_boxes = get_rois(fused, min_area=500, exclusion_regions=exclusion_regions, camera_id=camera_id)
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

    bridge.flush()
    cap.release()
    P2P3Bridge._finalize_track = original_finalize

    return bridge.get_completed_events()


def rebuild_with_threshold(raw_events, threshold_frac, apply_fallback, camera_id):
    """Given real raw events (with captured seat_id_counts), recompute
    seat_ids at the given threshold/fix-version, then re-run the rest of the
    real pipeline (enrich -> segment_events -> cluster -> link) exactly as
    production does, so ground-truth checks reflect the true end-to-end effect."""
    rebuilt = []
    for ev in raw_events:
        ev2 = dict(ev)
        seat_counts = ev.get('_captured_seat_id_counts', {})
        ev2['seat_ids'] = compute_dominant_seats(seat_counts, threshold_frac, apply_fallback)
        rebuilt.append(ev2)

    enriched = []
    for ev in rebuilt:
        intensities = ev.get('motion_intensities', [0.0])
        avg_mi = float(np.mean(intensities)) if intensities else 0.0
        peak_mi = float(max(intensities)) if intensities else 0.0
        mog2_ratios = ev.get('mog2_ratios', [0.0])
        avg_mog2 = float(np.mean(mog2_ratios)) if mog2_ratios else 0.0
        ms = {'avg_motion_intensity': avg_mi, 'peak_intensity': peak_mi, 'mog2_foreground_ratio': avg_mog2}
        enriched.append(enrich_event_with_motion_fields(ev, ms))

    confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
    clusters = cluster_incidents(confirmed, time_gap_sec=5.0, camera_id=camera_id)
    linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=60.0, camera_id=camera_id)
    return rebuilt, linked


def get_seats(ev):
    return sorted(ev.get("seat_ids", []))


def clip4_ground_truth(linked):
    pair1 = [ev for ev in linked if any(s in get_seats(ev) for s in ["seat_66", "seat_61"])
             and ev.get("start_time", 0) <= 15 and ev.get("end_time", 0) >= 2]
    pair2 = [ev for ev in linked if any(s in get_seats(ev) for s in ["seat_64", "seat_65"])
             and ev.get("start_time", 0) >= 60]
    pair2_ids = {ev["event_id"] for ev in pair2}
    pair2_links = [(ev["event_id"], ev.get("related_event_id")) for ev in pair2 if ev.get("related_event_id")]
    pair2_all_linked = bool(pair2_links) and all(rid in pair2_ids for _, rid in pair2_links)
    return {"pair1_present": len(pair1) > 0, "pair2_count": len(pair2), "pair2_cross_linked": pair2_all_linked}


def dahisar1_ground_truth(linked):
    adj_map = {'seat_1': {'seat_2'}, 'seat_2': {'seat_1','seat_3'}, 'seat_3': {'seat_2','seat_4'},
               'seat_4': {'seat_3','seat_5'}, 'seat_5': {'seat_4','seat_6'}, 'seat_6': {'seat_5'}}
    ev_by_id = {ev['event_id']: ev for ev in linked}
    adjacent_linked = 0
    for ev in linked:
        rel_id = ev.get('related_event_id')
        if not rel_id:
            continue
        rel_ev = ev_by_id.get(rel_id)
        if not rel_ev:
            continue
        sa, sb = set(get_seats(ev)), set(get_seats(rel_ev))
        if sa != sb and any(sb & adj_map.get(s, set()) for s in sa):
            adjacent_linked += 1
    seat7_leak = sum(1 for ev in linked if 'seat_7' in get_seats(ev) and ev.get('related_event_id'))
    return {"adjacent_seat_linked_pairs": adjacent_linked, "seat7_leakage": seat7_leak}


def main():
    all_results = {}
    for clip_key, cfg in CLIPS.items():
        print(f"\n########## {clip_key}: running real pipeline once (capturing seat_id_counts) ##########", flush=True)
        raw_events = run_pipeline_capture_seat_counts(cfg)
        print(f"  raw events: {len(raw_events)}", flush=True)
        with_counts = [e for e in raw_events if e.get('_captured_seat_id_counts')]
        print(f"  events with nonempty seat_id_counts: {len(with_counts)}", flush=True)

        results = {}
        for thr in [0.25, 0.30, 0.35]:
            for fix_label, apply_fb in [("OLD_no_fallback", False), ("NEW_with_fallback", True)]:
                rebuilt, linked = rebuild_with_threshold(raw_events, thr, apply_fb, cfg["camera_id"])
                empty_count = sum(1 for e in rebuilt if not e['seat_ids'])
                nonempty_count = len(rebuilt) - empty_count
                gt = clip4_ground_truth(linked) if clip_key == "clip4" else dahisar1_ground_truth(linked)
                results[(thr, fix_label)] = {
                    "empty_seat_ids_count": empty_count,
                    "nonempty_seat_ids_count": nonempty_count,
                    "confirmed_count": len(linked),
                    "ground_truth": gt,
                }
                print(f"  thr={thr} {fix_label}: empty={empty_count} nonempty={nonempty_count} confirmed={len(linked)} gt={gt}", flush=True)

        # --- Check 1: NEW == OLD at 0.25 (must be a true no-op) ---
        old_25, _ = rebuild_with_threshold(raw_events, 0.25, False, cfg["camera_id"])
        new_25, _ = rebuild_with_threshold(raw_events, 0.25, True, cfg["camera_id"])
        diffs_at_25 = [(o['event_id'], o['seat_ids'], n['seat_ids'])
                       for o, n in zip(old_25, new_25) if o['seat_ids'] != n['seat_ids']]
        print(f"\n  [{clip_key}] Events where NEW differs from OLD at threshold=0.25: {len(diffs_at_25)}")
        for row in diffs_at_25[:10]:
            print("   ", row)

        # --- Check 2: recovered cases at 0.30/0.35 ---
        for thr in [0.30, 0.35]:
            old, _ = rebuild_with_threshold(raw_events, thr, False, cfg["camera_id"])
            new, _ = rebuild_with_threshold(raw_events, thr, True, cfg["camera_id"])
            recovered = [(o['event_id'], n['seat_ids'], o.get('_captured_seat_id_counts', {}))
                         for o, n in zip(old, new) if not o['seat_ids'] and n['seat_ids']]
            print(f"  [{clip_key}] thr={thr}: events recovered by fallback (OLD empty -> NEW non-empty): {len(recovered)}")
            for eid, new_seats, counts in recovered[:8]:
                total = sum(counts.values()) if counts else 0
                frac_str = {s: f"{100*c/total:.0f}%" for s, c in counts.items()} if total else {}
                print(f"      {eid}: NEW seat_ids={new_seats}  raw_counts={counts}  fractions={frac_str}")

        all_results[clip_key] = {
            "sweep": {f"{thr}_{lbl}": v for (thr, lbl), v in results.items()},
            "diffs_at_0.25": len(diffs_at_25),
        }
        with open(f"dominant_seat_sweep_{clip_key}.json", "w") as f:
            json.dump(all_results[clip_key], f, indent=2, default=str)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
