import random

def generate_mock_tracks(n=60, clip_name="dev_clip_01.mp4", seed=42,
                          audio_path="../../data/sample_clip.wav"):
    random.seed(seed)
    tracks = []
    for i in range(n):
        n_frames = random.randint(5, 40)
        start = round(random.uniform(0, 5), 2)
        duration = round(random.uniform(0.5, 2.0), 2)
        positions = [(random.randint(0, 640), random.randint(0, 480)) for _ in range(n_frames)]
        boxes = [(x, y, random.randint(20, 80), random.randint(20, 80)) for (x, y) in positions]
        tracks.append({
            "event_id": f"E{i+1:03d}",
            "clip_name": clip_name,
            "positions": positions,
            "boxes": boxes,
            "start_time": start,
            "end_time": start + duration,
            "total_frames": n_frames,
            "motion_bursts": random.randint(1, 3),
            "audio_path": audio_path,
            "object_detected": random.random() < 0.15,
            "is_invigilator": random.random() < 0.1,
            "exam_phase": random.choice(["start", "mid", "end"]),
        })
    return tracks


if __name__ == "__main__":
    tracks = generate_mock_tracks(5)
    for t in tracks:
        print(t["event_id"], t["start_time"], t["end_time"], t["is_invigilator"], t["exam_phase"])