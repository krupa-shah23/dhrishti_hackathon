import cv2
import numpy as np
from src.motion.motion import MotionEstimator, fuse_motion_signal
from src.motion.grid_config import get_grid_config
from src.motion.exclusion_regions import get_exclusion_mask

def probe_intervention():
    video_path = r"data\drishti\07_seat_exchange.mkv"
    camera_id = "DAHISAR1"
    
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
    
    start_sec = 4406.0
    end_sec = 4415.0
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    estimator = MotionEstimator(history=500, var_threshold=16, detect_shadows=False)
    exclusion_mask = get_exclusion_mask(camera_id, orig_w, orig_h)
    
    frame_index = int(start_sec * source_fps)
    
    print(f"Probing frames from {start_sec}s to {end_sec}s...")
    
    distinct_cells = set()
    frames_with_motion = 0
    simultaneous_spikes = 0
    cell_history_list = []  # full per-frame sequence for is_invigilator_motion
    
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
                if intensity > 0.1:
                    active_cells.add(seat_id)
                    
        # Always append (including empty frames) to preserve temporal sequence
        cell_history_list.append(active_cells)

        if len(active_cells) > 0:
            print(f"[{timestamp_sec:.2f}s] Active cells: {active_cells}")
            frames_with_motion += 1
            distinct_cells.update(active_cells)
            if len(active_cells) > 2:
                simultaneous_spikes += 1
                
        frame_index += 1
        
    print(f"\nProbe Summary for {start_sec}s-{end_sec}s:")
    print(f"Frames with detected motion: {frames_with_motion}")
    print(f"Total distinct cells hit: {len(distinct_cells)} -> {distinct_cells}")
    print(f"Simultaneous spikes (>2 cells): {simultaneous_spikes}")

    # Call is_invigilator_motion directly on the collected history
    from src.motion.heuristics import is_invigilator_motion
    result = is_invigilator_motion(cell_history_list)
    print(f"\nis_invigilator_motion({start_sec}s-{end_sec}s) = {result}")
    if result:
        print("PASS: Guard intervention correctly detected.")
    else:
        print("FAIL: Guard intervention NOT detected — investigate heuristic.")

if __name__ == "__main__":
    probe_intervention()
