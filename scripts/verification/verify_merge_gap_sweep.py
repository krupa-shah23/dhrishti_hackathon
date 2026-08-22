"""
verify_merge_gap_sweep.py
Validates merge_gap_sec=2.0 (segment_events() Stage 2 same-seat merge) on clips
NOT used to fix/tune it (only clip 08 was used for that).

For each target clip:
  1. Run the pipeline, get Stage-1-gated events (motion+duration gate, no merge).
  2. Compute the actual merged output at merge_gap_sec=2.0 (current default).
  3. Independently replay the Stage-2 merge decision logic to record the *exact gap*
     between every pair of same-seat events that got merged -- so we can separate
     near-zero gaps (fragmentation glue, expected/good) from near-ceiling gaps
     (candidate over-merges, need frame-level scrutiny).
  4. Optionally sweep merge_gap_sec in {1.0, 1.5, 2.0, 2.5} and report how many
     merges occur / survive at each value, for any near-ceiling case found.

Usage: python scripts/verification/verify_merge_gap_sweep.py <clip4|dahisar1>
"""
import sys, json
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
        max_frames=39600,  # first 30 min @ 22fps, matches verify_dahisar1_clustering.py
        missing_threshold=30,
    ),
}


def run_pipeline(cfg):
    video = cfg["video"]
    camera_id = cfg["camera_id"]
    clip_name = cfg["clip_name"]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    run_frames = min(total_frames, cfg["max_frames"]) if cfg["max_frames"] else total_frames
    print(f"\nClip: {clip_name}, camera={camera_id}, FPS={fps:.2f}, total={total_frames}, "
          f"processing {run_frames} frames ({run_frames/fps:.1f}s)")

    motion_est = MotionEstimator(history=500, var_threshold=25, learning_rate=0.0008)
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

    print("[Running pipeline...]")
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
        if fi % 2000 == 0 and fi > 0:
            print(f"  ... frame {fi}/{run_frames} ({fi/fps:.1f}s)")

    bridge.flush()
    cap.release()
    P2P3Bridge._finalize_track = original_finalize

    all_events = bridge.get_completed_events()
    print(f"Total raw events from P2P3Bridge: {len(all_events)}")

    enriched = []
    for ev in all_events:
        intensities = ev.get('motion_intensities', [0.0])
        avg_mi = float(np.mean(intensities)) if intensities else 0.0
        peak_mi = float(max(intensities)) if intensities else 0.0
        ms = {'avg_motion_intensity': avg_mi, 'peak_intensity': peak_mi, 'mog2_foreground_ratio': 1.0}
        enriched.append(enrich_event_with_motion_fields(ev, ms))

    return enriched, fps


def find_merge_gaps(gated_events, merge_gap_sec):
    """
    Replays Stage-2 merge decision logic (same as event_clustering.segment_events)
    but records the (seat, gap, ev_a, ev_b) for every pair that gets merged,
    instead of just returning the merged list.
    """
    sorted_evs = sorted(gated_events, key=lambda e: e.get("start_time", 0.0))
    if not sorted_evs:
        return [], []

    merges = []  # list of dicts: gap, seats, a_event, b_event
    merged = []
    current = dict(sorted_evs[0])
    current["seat_ids"] = list(current.get("seat_ids", []))
    current_src = dict(sorted_evs[0])  # last raw sub-event absorbed, for gap reporting

    for ev in sorted_evs[1:]:
        seats_cur = set(current.get("seat_ids", []))
        seats_next = set(ev.get("seat_ids", []))
        gap = ev.get("start_time", 0.0) - current.get("end_time", 0.0)
        seats_overlap = bool(seats_cur & seats_next) or (not seats_cur and not seats_next)

        if seats_overlap and gap <= merge_gap_sec:
            merges.append({
                "gap": gap,
                "seats": sorted(seats_cur & seats_next) if (seats_cur & seats_next) else sorted(seats_cur | seats_next),
                "a_event_id": current_src.get("event_id"),
                "a_start": current_src.get("start_time"),
                "a_end": current_src.get("end_time"),
                "b_event_id": ev.get("event_id"),
                "b_start": ev.get("start_time"),
                "b_end": ev.get("end_time"),
            })
            current["end_time"] = max(current["end_time"], ev.get("end_time", 0.0))
            current["end_frame"] = max(current.get("end_frame", 0), ev.get("end_frame", 0))
            current["seat_ids"] = sorted(seats_cur | seats_next)
            current["avg_motion_intensity"] = max(current.get("avg_motion_intensity", 0.0), ev.get("avg_motion_intensity", 0.0))
            current["peak_intensity"] = max(current.get("peak_intensity", 0.0), ev.get("peak_intensity", 0.0))
            current_src = dict(ev)
        else:
            merged.append(current)
            current = dict(ev)
            current["seat_ids"] = list(current.get("seat_ids", []))
            current_src = dict(ev)

    merged.append(current)
    return merged, merges


def main():
    clip_key = sys.argv[1] if len(sys.argv) > 1 else "clip4"
    cfg = CLIPS[clip_key]

    enriched, fps = run_pipeline(cfg)

    # Stage-1 gate only (no merge) -- merge_gap_sec<=0 short-circuits merge in segment_events
    gated = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0, merge_gap_sec=0.0)
    print(f"\nAfter Stage-1 gate only: {len(gated)} events")

    print("\n=== MERGE-GAP SWEEP (Stage-2 merge decisions per merge_gap_sec value) ===")
    for mg in [1.0, 1.5, 2.0, 2.5]:
        merged, merges = find_merge_gaps(gated, mg)
        near_ceiling = [m for m in merges if m["gap"] > mg - 0.5]  # top 25% of window
        print(f"\nmerge_gap_sec={mg}: {len(gated)} gated -> {len(merged)} merged "
              f"({len(merges)} merge operations)")
        if merges:
            gaps = sorted(m["gap"] for m in merges)
            print(f"  gap distribution: min={gaps[0]:.3f}s max={gaps[-1]:.3f}s "
                  f"median={gaps[len(gaps)//2]:.3f}s")
        if near_ceiling:
            print(f"  NEAR-CEILING merges (gap > {mg-0.5:.1f}s): {len(near_ceiling)}")
            for m in near_ceiling:
                print(f"    seats={m['seats']} gap={m['gap']:.3f}s : "
                      f"{m['a_event_id']}[{m['a_start']:.2f}-{m['a_end']:.2f}s] <-> "
                      f"{m['b_event_id']}[{m['b_start']:.2f}-{m['b_end']:.2f}s]")

    # Detailed report at the current default (2.0s)
    merged_2s, merges_2s = find_merge_gaps(gated, 2.0)
    print(f"\n=== DETAIL @ merge_gap_sec=2.0 (current default) ===")
    print(f"Total merge operations: {len(merges_2s)}")
    for m in sorted(merges_2s, key=lambda x: -x["gap"]):
        flag = "  <-- NEAR CEILING (>1.5s), needs frame check" if m["gap"] > 1.5 else ""
        print(f"  seats={m['seats']} gap={m['gap']:.3f}s : "
              f"{m['a_event_id']}[{m['a_start']:.2f}-{m['a_end']:.2f}s] <-> "
              f"{m['b_event_id']}[{m['b_start']:.2f}-{m['b_end']:.2f}s]{flag}")

    # Final pipeline output at default 2.0s (for ground-truth comparison)
    confirmed = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)  # default merge_gap_sec=2.0
    clusters = cluster_incidents(confirmed, time_gap_sec=5.0, camera_id=cfg["camera_id"])
    linked = link_related_events(clusters, spatial_radius_px=150.0, time_window_sec=60.0, camera_id=cfg["camera_id"])
    print(f"\n=== FINAL PIPELINE OUTPUT @ merge_gap_sec=2.0 ===")
    print(f"confirmed={len(confirmed)} clusters={len(clusters)} linked_events={len(linked)}")

    out = {
        "clip": clip_key,
        "gated_count": len(gated),
        "merges_at_2s": merges_2s,
        "final_confirmed": len(confirmed),
        "final_clusters": len(clusters),
    }
    with open(f"merge_gap_report_{clip_key}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote merge_gap_report_{clip_key}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
