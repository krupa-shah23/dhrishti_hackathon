import cv2
import json
import time
import psutil
import os
from pathlib import Path
from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.p2_p3_bridge import P2P3Bridge
from src.integration.event_clustering import enrich_event_with_motion_fields, segment_events, cluster_incidents
import sys

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_video(video_path, camera_id=None):
    print(f"\n{'='*50}\nProcessing: {video_path} (Camera: {camera_id})")
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    duration = total_frames / fps if fps > 0 else 0
    print(f"Duration: {duration:.2f}s")
    
    pipeline = P1P2TrackerPipeline(fps=fps)
    bridge = P2P3Bridge(missing_threshold=15, fps=fps, camera_id=camera_id)

    mem_start = get_memory_usage()
    start_time = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, frame_index=frame_idx, camera_id=camera_id)
        bridge.process_fused_tracks(
            result["fused_tracks"],
            frame_index=frame_idx,
            pose_signals=result.get("pose_signals"),
            motion_intensity=result.get("motion_intensity"),
            mog2_foreground_ratio=result.get("mog2_foreground_ratio"),
        )
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames. Current Memory: {get_memory_usage():.2f} MB")

    # Finalize
    bridge.flush()
    raw_events = bridge.get_completed_events()

    # Enrichment: uses event["motion_intensities"]/["mog2_ratios"] (genuine
    # per-frame measurements accumulated above) via enrich_event_with_motion_fields()'s
    # own fallback -- also sets invigilator_excluded/event_type, which the
    # hand-rolled version this replaced did not, silently skipping the
    # invigilator-suppression gate in segment_events() below.
    enriched = [enrich_event_with_motion_fields(ev) for ev in raw_events]

    # Segment & Cluster
    segmented = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
    # cluster_incidents() returns List[List[event]] (clusters of events), not a
    # flat list -- flatten before treating entries as individual event dicts
    # below (the code this replaced iterated `clustered` as if flat, which
    # would AttributeError the moment it hit a real, non-empty cluster).
    clusters = cluster_incidents(segmented, camera_id=camera_id if camera_id else "Camera12")
    clustered = [ev for cluster in clusters for ev in cluster]

    end_time = time.time()
    mem_end = get_memory_usage()
    
    print(f"\nResults for {os.path.basename(video_path)}:")
    print(f"Frames Processed: {frame_idx}")
    print(f"Processing Time: {end_time - start_time:.2f}s")
    print(f"Memory (Start -> End): {mem_start:.2f} MB -> {mem_end:.2f} MB")
    print(f"Track Count (Raw Events): {len(raw_events)}")
    print(f"Segmented Events: {len(segmented)}")
    print(f"Clustered Incidents: {len(clustered)}")
    
    # Output events to JSON for inspection (repo-relative, not a local machine path)
    out_dir = os.path.join("outputs", "verification")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"phase4_{os.path.basename(video_path)}.json")
    with open(out_file, 'w') as f:
        json.dump(clustered, f, indent=2)
    print(f"Events saved to {out_file}")

    # Validation checks. seat_ids never literally contains "unknown" (see
    # p2_p3_bridge.py's seat_id_counts, which only tracks non-"unknown"
    # seats) -- an empty seat_ids list is what an unconfigured-camera/no-seat
    # track actually looks like.
    unknown_seat_count = sum(1 for e in clustered if not e.get('seat_ids'))
    invigilator_excluded = sum(1 for e in raw_events if e.get('is_invigilator', False))
    print(f"Unknown seat events: {unknown_seat_count}")
    print(f"Excluded invigilator events: {invigilator_excluded}")

if __name__ == '__main__':
    videos_to_test = [
        ("data/drishti/04_candidate_talking.mkv", "Camera12"),
        ("data/drishti/08_seat12_copying.mkv", "Camera12"),
        ("data/drishti/01_phone_use.mkv", None)  # Unconfigured camera
    ]
    
    for vid, cam in videos_to_test:
        run_video(vid, cam)
