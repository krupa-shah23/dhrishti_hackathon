import sys, os
sys.path.insert(0, os.getcwd())

from src.integration.p1_p2_tracker import process_video_live

for result in process_video_live('data/college_dataset/01.Candidate was found using a mobile phone in the examination hall..mkv', clip_name='clip01'):
    fi = result['frame_index']
    if 140 <= fi <= 160 or 295 <= fi <= 410:
        print(f"frame {fi} ({fi/25:.1f}s): {len(result['rois'])} roi box(es) -> {result['rois']}")
