"""
verify_phase_aware_alpha_sweep.py
§3.2 phase-aware alpha follow-up: the steady-phase learning-rate value (0.0008,
currently a functional no-op vs. the original fixed rate) was only validated on
one clip against one incident pair, with 3 candidate values all 0.0002 apart.

Widens the sweep to steady_learning_rate in {0.0004, 0.0008, 0.0015, 0.0025}
(meaningfully different, not just +/-0.0002 around 0.0008) across two real
clips with established ground truth: Camera12 clip 4 (Pair 1 / Pair 2) and
DAHISAR1 clip 07 (seat-exchange adjacency, 30-min window -- same window used
throughout this session's other DAHISAR1 verification work).

For each (clip, alpha) combination, records: raw/gated/confirmed/cluster
counts, and the same ground-truth checks already used to validate these clips
elsewhere this session.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from src.integration.event_clustering import segment_events, enrich_event_with_motion_fields, cluster_incidents, link_related_events
from src.integration.p2_p3_bridge import P2P3Bridge
from src.motion.motion import MotionEstimator, fuse_motion_signal, motion_intensity as _mi
from src.motion.roi import get_rois
from src.motion.exclusion_regions import get_exclusion_regions, CAMERA_CONFIGS
from src.motion.grid_config import GRID_CONFIGS
from src.track_det.tracker import track, reset_tracker, get_track_history
from src.track_det.fusion import fuse_track_detections
from src.track_det.invigilator_filter import is_invigilator_track
import cv2

ALPHAS = [0.0004, 0.0008, 0.0015, 0.0025]

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
        max_frames=39600,  # first 30 min @ 22fps, same window as other DAHISAR1 checks this session
        missing_threshold=30,
    ),
}


def run_pipeline(cfg, steady_learning_rate):
    video = cfg["video"]
    camera_id = cfg["camera_id"]
    clip_name = cfg["clip_name"]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    run_frames = min(total_frames, cfg["max_frames"]) if cfg["max_frames"] else total_frames

    motion_est = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008,
                                  steady_learning_rate=steady_learning_rate)
    exclusion_regions = get_exclusion_regions(clip_name)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_key = (camera_id, (frame_w, frame_h))
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
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (fx2-fx1)*(fy2-fy1) + (bx2-bx1)*(by2-by1) - inter
                    iou = inter/union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou, best_seat = iou, sid
            ft['seat_id'] = best_seat
        bridge.process_fused_tracks(fused_tracks, fi, motion_intensity=mi)

    bridge.flush()
    cap.release()
    P2P3Bridge._finalize_track = original_finalize

    all_events = bridge.get_completed_events()

    enriched = []
    for ev in all_events:
        intensities = ev.get('motion_intensities', [0.0])
        avg_mi = float(np.mean(intensities)) if intensities else 0.0
        peak_mi = float(max(intensities)) if intensities else 0.0
        ms = {'avg_motion_intensity': avg_mi, 'peak_intensity': peak_mi, 'mog2_foreground_ratio': 1.0}
        enriched.append(enrich_event_with_motion_fields(ev, ms))

    confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
    clusters = cluster_incidents(confirmed, time_gap_sec=5.0, camera_id=camera_id)
    linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=60.0, camera_id=camera_id)

    return dict(raw=len(all_events), confirmed=len(confirmed), clusters=len(clusters), linked=linked)


def get_seats(ev):
    return sorted(ev.get("seat_ids", []))


def check_clip4_ground_truth(linked):
    pair1 = [ev for ev in linked
             if any(s in get_seats(ev) for s in ["seat_66", "seat_61"])
             and ev.get("start_time", 0) <= 15 and ev.get("end_time", 0) >= 2]
    pair2 = [ev for ev in linked
             if any(s in get_seats(ev) for s in ["seat_64", "seat_65"])
             and ev.get("start_time", 0) >= 60]
    pair2_ids = {ev["event_id"] for ev in pair2}
    pair2_links = [(ev["event_id"], ev.get("related_event_id")) for ev in pair2 if ev.get("related_event_id")]
    pair2_all_linked = bool(pair2_links) and all(rid in pair2_ids for _, rid in pair2_links)
    return {
        "pair1_event_count": len(pair1),
        "pair1_present": len(pair1) > 0,
        "pair2_event_count": len(pair2),
        "pair2_cross_linked": pair2_all_linked,
    }


def check_dahisar1_ground_truth(linked):
    adj_map = {
        'seat_1': {'seat_2'}, 'seat_2': {'seat_1', 'seat_3'},
        'seat_3': {'seat_2', 'seat_4'}, 'seat_4': {'seat_3', 'seat_5'},
        'seat_5': {'seat_4', 'seat_6'}, 'seat_6': {'seat_5'},
    }
    ev_by_id = {ev['event_id']: ev for ev in linked}
    adjacent_linked_pairs = 0
    for ev in linked:
        rel_id = ev.get('related_event_id')
        if not rel_id:
            continue
        rel_ev = ev_by_id.get(rel_id)
        if not rel_ev:
            continue
        sa, sb = set(get_seats(ev)), set(get_seats(rel_ev))
        if sa != sb and any(sb & adj_map.get(s, set()) for s in sa):
            adjacent_linked_pairs += 1
    seat7_linked = sum(1 for ev in linked if 'seat_7' in get_seats(ev) and ev.get('related_event_id'))
    return {
        "adjacent_seat_linked_pairs": adjacent_linked_pairs,
        "seat7_leakage": seat7_linked,
    }


def main():
    results = []
    for clip_key, cfg in CLIPS.items():
        for alpha in ALPHAS:
            t0 = time.time()
            print(f"\n=== {clip_key} steady_learning_rate={alpha} ===", flush=True)
            out = run_pipeline(cfg, alpha)
            elapsed = time.time() - t0
            row = {
                "clip": clip_key,
                "alpha": alpha,
                "raw": out["raw"],
                "confirmed": out["confirmed"],
                "clusters": out["clusters"],
                "elapsed_sec": round(elapsed, 1),
            }
            if clip_key == "clip4":
                row["ground_truth"] = check_clip4_ground_truth(out["linked"])
            else:
                row["ground_truth"] = check_dahisar1_ground_truth(out["linked"])
            print(json.dumps(row, indent=2), flush=True)
            results.append(row)
            # write incrementally so progress is visible mid-run
            with open("phase_aware_alpha_sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\n\n=== FULL SWEEP TABLE ===")
    print(f"{'clip':10} {'alpha':8} {'raw':6} {'confirmed':10} {'clusters':9} ground_truth")
    for r in results:
        print(f"{r['clip']:10} {r['alpha']:<8} {r['raw']:6} {r['confirmed']:10} {r['clusters']:9} {r['ground_truth']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
