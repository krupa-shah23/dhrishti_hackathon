import numpy as np
from audio_utils import get_audio_energy, get_onset_strength

def extract_features(event: dict) -> dict:
    duration = event.get("duration", 0.0)
    motion_area = event.get("avg_motion_intensity", 0.0)
    roi_size = event.get("peak_intensity", 0.0)
    density = float(motion_area / roi_size) if roi_size > 0 else 0.0
    frequency = event.get("repetition_count", 1)
    mog2_ratio = event.get("mog2_foreground_ratio", 0.0)

    audio_energy = 0.0
    onset_strength = 0.0
    audio_path = event.get("audio_path")
    if audio_path:
        audio_energy = get_audio_energy(audio_path, event["start_time"], event["end_time"])
        onset_strength = get_onset_strength(audio_path, event["start_time"], event["end_time"])

    object_flag = int(event.get("object_detected") is not None)
    phase = event.get("exam_phase", "mid")
    phase_start = 1 if phase == "start" else 0
    phase_end = 1 if phase == "end" else 0

    feats = {
        "motion_area": motion_area,
        "duration": duration,
        "frequency": frequency,
        "roi_size": roi_size,
        "density": density,
        "mog2_foreground_ratio": mog2_ratio,
        "audio_energy": audio_energy,
        "onset_strength": onset_strength,
        "object_flag": object_flag,
        "phase_start": phase_start,
        "phase_end": phase_end,
    }
    for k, v in feats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feats[k] = 0.0
    return feats

if __name__ == "__main__":
    from mock_event_data import generate_mock_events
    events = generate_mock_events(3)
    for e in events:
        print(e["event_id"], extract_features(e))