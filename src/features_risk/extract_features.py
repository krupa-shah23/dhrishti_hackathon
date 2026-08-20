import numpy as np
from audio_utils import get_audio_energy, get_onset_strength


def extract_features(track: dict) -> dict:
    positions = track.get("positions") or []
    boxes = track.get("boxes") or []
    start = track.get("start_time", 0.0) or 0.0
    end = track.get("end_time", 0.0) or 0.0
    duration = max(end - start, 0.0)

    areas = [w * h for (_, _, w, h) in boxes] if boxes else [0.0]
    motion_area = float(sum(areas) / len(areas)) if areas else 0.0

    speeds = []
    for i in range(1, len(positions)):
        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        speeds.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

    if len(positions) >= 2:
        x1, y1 = positions[0]
        x2, y2 = positions[-1]
        direction = float(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))) if (x2 != x1 or y2 != y1) else 0.0
    else:
        direction = 0.0

    roi_size = float(max(areas)) if areas else 0.0
    density = float(motion_area / roi_size) if roi_size > 0 else 0.0
    total_frames = track.get("total_frames") or max(len(positions), 1)
    persistence = float(len(positions) / total_frames) if total_frames > 0 else 0.0
    frequency = int(track.get("motion_bursts") or 1)

    audio_energy = 0.0
    onset_strength = 0.0
    if track.get("audio_path"):
        audio_energy = get_audio_energy(track["audio_path"], start, end)
        onset_strength = get_onset_strength(track["audio_path"], start, end)

    object_flag = int(bool(track.get("object_detected", False)))

    # NEW: P2's invigilator filter (track-local, no re-id needed)
    invigilator_flag = int(bool(track.get("is_invigilator", False)))

    # NEW: P4's exam phase bucket (categorical -> one-hot)
    phase = track.get("exam_phase", "mid")
    phase_start = 1 if phase == "start" else 0
    phase_end = 1 if phase == "end" else 0

    feats = {
        "motion_area": motion_area,
        "duration": duration,
        "frequency": frequency,
        "speed": speed,
        "roi_size": roi_size,
        "direction": direction,
        "density": density,
        "persistence": persistence,
        "audio_energy": audio_energy,
        "onset_strength": onset_strength,
        "object_flag": object_flag,
        "invigilator_flag": invigilator_flag,
        "phase_start": phase_start,
        "phase_end": phase_end,
    }

    for k, v in feats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feats[k] = 0.0

    return feats


if __name__ == "__main__":
    from mock_track_data import generate_mock_tracks
    tracks = generate_mock_tracks(3)
    for t in tracks:
        print(t["event_id"], extract_features(t))