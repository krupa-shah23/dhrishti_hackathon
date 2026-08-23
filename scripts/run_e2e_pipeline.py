import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.integration.p1_p2_tracker import P1P2TrackerPipeline
from src.integration.p2_p3_bridge import P2P3Bridge
from src.features_risk.extract_features import extract_features
from src.integration.p3_p4_bridge import p3_event_to_csv_row, p3_events_to_timeline_arrays
from src.outputs_eval.logger import init_log, log_event
from src.outputs_eval.heatmap import accumulate_mask, generate_heatmap, save_heatmap
from src.outputs_eval.timeline import plot_timeline
from src.outputs_eval.report import generate_report

def run_e2e():
    video_path = r"C:\Users\prisha\OneDrive\Pictures\Desktop\drishti-p4\data\demo_clips\raw\sample_exam.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    csv_path = "outputs/e2e_events.csv"
    heatmap_path = "outputs/e2e_heatmap.png"
    timeline_path = "outputs/e2e_timeline.png"
    report_path = "outputs/e2e_report.pdf"
    
    os.makedirs("outputs", exist_ok=True)
    init_log(csv_path)

    pipeline = P1P2TrackerPipeline()
    p2p3_bridge = P2P3Bridge(fps=fps)
    
    accumulator = np.zeros((height, width), dtype=np.float32)
    events_with_scores = []
    
    print("Starting E2E Pipeline Processing...")
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # P1 & P2
        result = pipeline.process_frame(frame, frame_index)
        
        # P4 Heatmap - grab mask from the motion estimator
        mask = pipeline.motion_estimator.get_motion_mask(frame)
        accumulator = accumulate_mask(accumulator, mask)
        
        # P2->P3 Bridge
        fused_tracks = result["fused_tracks"]
        p2p3_bridge.process_fused_tracks(fused_tracks, frame_index)
        
        # Check for completed events
        completed_events = p2p3_bridge.get_completed_events()
        for event in completed_events:
            # P3 Feature Extraction
            features = extract_features(event)
            # Mock risk score since xgboost is blocked
            risk_score = 0.5 + 0.1 * features.get("object_flag", 0) + 0.05 * (features.get("motion_area", 0) / 10000)
            risk_score = min(1.0, max(0.0, risk_score))
            
            # Save for timeline plot
            events_with_scores.append((event, features, risk_score))
            
            # P4 CSV Logging
            row = p3_event_to_csv_row(event, features, risk_score, fps, width, height)
            log_event(csv_path, row)
            
        if frame_index % 100 == 0:
            print(f"Processed frame {frame_index}...")
            
        frame_index += 1

    cap.release()
    
    # Flush remaining tracks
    p2p3_bridge.flush()
    for event in p2p3_bridge.get_completed_events():
        features = extract_features(event)
        risk_score = 0.5 + 0.1 * features.get("object_flag", 0) + 0.05 * (features.get("motion_area", 0) / 10000)
        risk_score = min(1.0, max(0.0, risk_score))
        events_with_scores.append((event, features, risk_score))
        row = p3_event_to_csv_row(event, features, risk_score, fps, width, height)
        log_event(csv_path, row)
        
    print("Processing complete. Generating outputs...")
    
    # P4 Heatmap Generation
    heatmap = generate_heatmap(accumulator)
    save_heatmap(heatmap, heatmap_path)
    
    # P4 Timeline Generation
    timestamps, motion_scores, risk_scores = p3_events_to_timeline_arrays(
        events_with_scores, fps, width, height
    )
    plot_timeline(timestamps, motion_scores, risk_scores, timeline_path, video_id="sample_exam")
    
    # P4 Report Generation
    generate_report(
        video_path=video_path,
        csv_path=csv_path,
        heatmap_path=heatmap_path,
        timeline_path=timeline_path,
        output_path=report_path,
        processing_time_s=10.0, # dummy value
        risk_threshold=0.65
    )
    print(f"E2E Run completed. All outputs saved in 'outputs/' directory.")

if __name__ == "__main__":
    run_e2e()
