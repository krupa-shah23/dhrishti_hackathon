import cv2
import json
import time
import psutil
import os
from pathlib import Path
from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.event_clustering import segment_events, cluster_incidents
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
    
    mem_start = get_memory_usage()
    start_time = time.time()
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        pipeline.process_frame(frame, frame_index=frame_idx, camera_id=camera_id)
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames. Current Memory: {get_memory_usage():.2f} MB")
            
    # Finalize
    for tid in list(pipeline.bridge.active_tracks.keys()):
        pipeline.bridge._finalize_track(tid)
        
    raw_events = pipeline.bridge.get_completed_events()
    
    # Enrichment
    enriched = []
    for ev in raw_events:
        # Phase 3 contract simulation for motion stats (using genuine metrics)
        intensities = ev.get("motion_intensities", [0.0])
        mog2_ratios = ev.get("mog2_ratios", [0.0])
        avg_mi = sum(intensities)/len(intensities) if intensities else 0.0
        peak_mi = max(intensities) if intensities else 0.0
        avg_mog2 = sum(mog2_ratios)/len(mog2_ratios) if mog2_ratios else 0.0
        
        ev["avg_motion_intensity"] = avg_mi
        ev["peak_intensity"] = peak_mi
        ev["mog2_foreground_ratio"] = avg_mog2
        enriched.append(ev)
        
    # Segment & Cluster
    segmented = segment_events(enriched, motion_threshold=0.02, min_duration_sec=1.0)
    clustered = cluster_incidents(segmented)
    
    end_time = time.time()
    mem_end = get_memory_usage()
    
    print(f"\nResults for {os.path.basename(video_path)}:")
    print(f"Frames Processed: {frame_idx}")
    print(f"Processing Time: {end_time - start_time:.2f}s")
    print(f"Memory (Start -> End): {mem_start:.2f} MB -> {mem_end:.2f} MB")
    print(f"Track Count (Raw Events): {len(raw_events)}")
    print(f"Segmented Events: {len(segmented)}")
    print(f"Clustered Incidents: {len(clustered)}")
    
    # Output events to JSON for inspection
    out_file = f"C:/Users/l/.gemini/antigravity-ide/brain/c2b13eb5-eb90-4ba1-8cf3-e34adf4fad4b/phase4_{os.path.basename(video_path)}.json"
    with open(out_file, 'w') as f:
        json.dump(clustered, f, indent=2)
    print(f"Events saved to {out_file}")
    
    # Validation checks
    unknown_seat_count = sum(1 for e in clustered if 'unknown' in e.get('seat_ids', []))
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
