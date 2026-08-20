import json

def load_real_tracks(json_path, audio_path=None):
    with open(json_path) as f:
        raw = json.load(f)

    tracks = []
    for t in raw:
        tracks.append({
            "event_id": f"T{t['track_id']:04d}",
            "clip_name": json_path,
            "positions": t.get("positions", []),
            "boxes": t.get("boxes", []),
            "start_time": t.get("start_time", 0),
            "end_time": t.get("end_time", 0),
            "total_frames": t.get("total_frames", 1),
            "motion_bursts": 1,
            "audio_path": audio_path,          # OEP likely has no separate audio file per track
            "object_detected": False,          # per P2: detector unreliable on OEP, exclude/flag
            "is_invigilator": False,           # not available yet, default
            "exam_phase": "mid",               # not available yet, default
        })
    return tracks


if __name__ == "__main__":
    tracks = load_real_tracks("../../data/sample_tracks_subject1.json")
    print(f"Loaded {len(tracks)} real tracks")
    print(tracks[0])