import cv2
import os
import argparse
from src.motion.grid_config import get_grid_config

def extract_and_draw(video_path, camera_id, time_sec, label, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    grid = get_grid_config(camera_id, width, height)
    seats = grid.get('seats', {})
    
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {time_sec}s")
        cap.release()
        return
        
    for sid, box in seats.items():
        x1, y1, x2, y2 = box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, sid, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    out_path = os.path.join(out_dir, f'{camera_id.lower()}_t{time_sec}_{label}.jpg')
    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path}")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and overlay seat grid on video frame")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--camera", required=True, help="Camera ID (e.g. DAHISAR1)")
    parser.add_argument("--time", type=float, required=True, help="Time in seconds")
    parser.add_argument("--label", default="frame", help="Label for output filename")
    parser.add_argument("--outdir", default=".", help="Output directory")
    
    args = parser.parse_args()
    extract_and_draw(args.video, args.camera, args.time, args.label, args.outdir)
