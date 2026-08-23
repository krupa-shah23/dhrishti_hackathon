import os
import sys
import random

from src.outputs_eval.logger import init_log, log_event, format_timestamp, format_roi_box
from src.outputs_eval.timeline import plot_timeline
from src.outputs_eval.report import generate_report

def run_report_test():
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/test_events.csv"
    timeline_path = "outputs/test_timeline.png"
    heatmap_path = "outputs/test_heatmap.png"  # Generated from previous test
    report_path = "outputs/test_report.pdf"
    
    # 1. Generate CSV data
    init_log(csv_path)
    timestamps = []
    motion_scores = []
    risk_scores = []
    
    fps = 25.0
    for i in range(100):
        frame_index = i * 10
        ts_str = format_timestamp(frame_index, fps)
        
        # Simulate some data
        motion_score = random.uniform(5.0, 95.0)
        risk_score = random.uniform(0.1, 0.9)
        roi_box = format_roi_box((100 + i, 100, 50, 50))
        confidence = random.uniform(0.6, 0.99)
        
        timestamps.append(frame_index / fps)
        motion_scores.append(motion_score)
        risk_scores.append(risk_score)
        
        event_data = {
            "timestamp": ts_str,
            "frame_index": frame_index,
            "roi_box": roi_box,
            "motion_score": motion_score,
            "audio_level": random.uniform(-40, 0),
            "risk_score": risk_score,
            "confidence": confidence
        }
        log_event(csv_path, event_data)
        
    print(f"Generated 100 events to {csv_path}")
    
    # 2. Generate Timeline Plot
    plot_timeline(
        timestamps=timestamps,
        motion_scores=motion_scores,
        risk_scores=risk_scores,
        output_path=timeline_path,
        video_id="test_video_123"
    )
    
    # 3. Generate PDF Report
    generate_report(
        video_path="fake_path/test_video_123.mp4",
        csv_path=csv_path,
        heatmap_path=heatmap_path,
        timeline_path=timeline_path,
        output_path=report_path,
        extra_metrics={"mock_f1": 0.88, "mock_iou": 0.75},
        processing_time_s=12.5,
        risk_threshold=0.65
    )

if __name__ == "__main__":
    run_report_test()
