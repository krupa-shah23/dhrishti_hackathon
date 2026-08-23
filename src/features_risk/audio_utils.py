import librosa
import numpy as np


def get_audio_energy(audio_path: str, start: float, end: float) -> float:
    duration = max(end - start, 0.01)
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=start, duration=duration)
        if y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=y)
        return float(np.mean(rms))
    except Exception as e:
        print(f"[audio_utils] energy fallback due to: {e}")
        return 0.0


def get_onset_strength(audio_path: str, start: float, end: float) -> float:
    duration = max(end - start, 0.01)
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=start, duration=duration)
        if y.size == 0:
            return 0.0
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return float(np.mean(onset_env))
    except Exception as e:
        print(f"[audio_utils] onset fallback due to: {e}")
        return 0.0


if __name__ == "__main__":
    path = "../../data/sample_clip.wav"
    print("energy:", get_audio_energy(path, 0, 2))
    print("onset:", get_onset_strength(path, 0, 2))