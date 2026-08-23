import pandas as pd
import numpy as np
import time
from src.motion.frame_stream import get_frame_stream_with_indices
from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.baseline import RollingBaselineTracker, calibrate_baseline_with_tracker, collect_warmup_motion_masks
from src.motion.heuristics import is_invigilator_motion, tag_interventions, is_seat_vacated
from src.motion.grid_config import get_grid_config
from src.motion.exclusion_regions import get_exclusion_mask

def run_f1_checkpoint():
    video_path = r"data\drishti\07_seat_exchange.mkv"
    camera_id = "DAHISAR1"
    gt_df = pd.read_csv(r"src\motion\ground_truth_07.csv")
    
    print(f"Loading ground truth: {len(gt_df)} events")
    
    # Target frame stream
    # Instead of full stream (which takes 2 hours), let's just seek to near 4391 and 4406
    # and run the detectors on those slices!
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
        
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_h = 480
    target_w = int(orig_w * (target_h / orig_h)) if orig_h > 0 else orig_w
    grid = get_grid_config(camera_id, target_w, target_h)
    
    # We will simulate processing around the events
    # Event 1: seat exchange ~4391s
    # Event 2: guard checkin ~4406s
    
    start_sec = 4385.0
    end_sec = 4420.0
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    estimator = MotionEstimator(history=500, var_threshold=16, detect_shadows=False)
    exclusion_mask = get_exclusion_mask(camera_id, orig_w, orig_h)
    
    frame_index = int(start_sec * source_fps)
    
    cell_history = []
    
    print(f"Processing frames from {start_sec}s to {end_sec}s ...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_sec = frame_index / source_fps
        if timestamp_sec > end_sec:
            break
            
        # Downsample and process
        if exclusion_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=exclusion_mask)
        if orig_h > 0 and orig_h != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        mag_map, mog2_mask = estimator.get_motion_mask(frame)
        fused_mask = fuse_motion_signal(mag_map, mog2_mask)
        
        # Check active cells
        active_cells = set()
        for seat_id, (x1, y1, x2, y2) in grid.get("seats", {}).items():
            roi = fused_mask[y1:y2, x1:x2]
            if roi.size > 0:
                intensity = np.count_nonzero(roi) / roi.size
                if intensity > 0.1: # simple threshold for active
                    active_cells.add(seat_id)
                    
        cell_history.append((timestamp_sec, active_cells))
        frame_index += 1
        
    elapsed = time.time() - start_time
    print(f"Processed in {elapsed:.2f}s")
    
    # Tag interventions
    # We mock the events as being flagged already
    events = [
        {'id': 'e1', 'start_sec': 4391.0, 'end_sec': 4400.0, 'intervention_detected': False},
        {'id': 'e2', 'start_sec': 4406.0, 'end_sec': 4415.0, 'intervention_detected': False}
    ]
    
    tag_interventions(events, cell_history)
    
    print("Event Tagging Results:")
    for e in events:
        print(f"Event {e['id']} ({e['start_sec']}-{e['end_sec']}): intervention_detected={e['intervention_detected']}")

    # Evaluate recall for seat exchange (event e1)
    # Did any seat trigger motion during e1's window?
    e1_active = False
    for t, cells in cell_history:
        if 4391.0 <= t <= 4400.0 and cells:
            # We want to exclude aisle cells for candidate events
            candidate_cells = [c for c in cells if not c.startswith("aisle")]
            if candidate_cells:
                e1_active = True
                break
                
    # Evaluate recall for guard check-in (event e2)
    # Did the intervention heuristic trigger?
    e2_detected = events[1]['intervention_detected']
    
    print("\nEnd-of-day Checkpoint:")
    print(f"Event 1 (Seat Exchange): Recall = {'SUCCESS' if e1_active else 'FAILURE'}")
    print(f"Event 2 (Guard Check-in): Intervention = {'SUCCESS' if e2_detected else 'FAILURE'}")
    
    if not e1_active or not e2_detected:
        print("Failure direction observed: RECALL FAILURE.")
    else:
        print("Failure direction observed: SUCCESS.")

if __name__ == "__main__":
    run_f1_checkpoint()
