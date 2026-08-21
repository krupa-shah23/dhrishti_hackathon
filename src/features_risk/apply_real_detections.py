from load_real_events import merge_real_detections_into_gt

merge_real_detections_into_gt(
    "../../data/ground_truth_labels.csv",
    "../../data/p3_detection_output_clip3.csv",
    video_id="clip3"
)