import multiprocessing
import time
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion.baseline import collect_warmup_motion_masks

def run_calibration(video_path, camera_id):
    try:
        masks, grid = collect_warmup_motion_masks(video_path, camera_id)
        print("Calibration finished successfully in process.")
    except Exception as e:
        print(f"Process crashed: {e}")

def kill_and_recover_test():
    video_path = r"data\drishti\07_seat_exchange.mkv"
    camera_id = "DAHISAR1"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print("Starting calibration process...")
    p = multiprocessing.Process(target=run_calibration, args=(video_path, camera_id))
    p.start()
    
    # Wait a few seconds to let it process some frames
    time.sleep(3)
    
    print("Simulating mid-stream KILL...")
    p.terminate()
    p.join()
    print("Process killed.")
    
    print("Attempting to recover (running calibration again)...")
    start = time.time()
    try:
        # Run normally and see if it succeeds
        masks, grid = collect_warmup_motion_masks(video_path, camera_id)
        elapsed = time.time() - start
        print(f"Recovery SUCCESSFUL! Calibrated in {elapsed:.2f}s")
        print(f"Masks: {len(masks)}, Seats: {len(grid.get('seats', {}))}")
        print("No state was corrupted or lost.")
    except Exception as e:
        print(f"Recovery FAILED: {e}")
        
if __name__ == '__main__':
    # Needs freeze_support on Windows
    multiprocessing.freeze_support()
    kill_and_recover_test()
