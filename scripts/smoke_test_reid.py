import sys, os
sys.path.insert(0, os.getcwd())

from src.integration.p1_p2_tracker import process_video_live

frame_count = 0
detection_count = 0
for result in process_video_live('data/college_dataset/01.Candidate was found using a mobile phone in the examination hall..mkv', clip_name='clip01'):
    for ft in result['fused_tracks']:
        if ft.get('class'):
            detection_count += 1
            print(f"frame {result['frame_index']} ({result['frame_index']/25:.1f}s): track_id={ft['track_id']} person_id={ft.get('person_id')} class={ft['class']} conf={ft['confidence']:.2f}")
    frame_count += 1

print(f"DONE - {frame_count} frames processed, {detection_count} total detections, no crash")
