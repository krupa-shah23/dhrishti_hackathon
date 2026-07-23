import librosa
import numpy as np


def get_audio_energy(audio_path: str, start: float, end: float) -> float:
    """Return mean RMS energy of audio between start and end (seconds)."""
    duration = max(end - start, 0.01)
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=start, duration=duration)
        if y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=y)
        return float(np.mean(rms))
    except Exception as e:
        print(f"[audio_utils] fallback due to: {e}")
        return 0.0


if __name__ == "__main__":
    val = get_audio_energy("../../data/sample_clip.wav", 0, 2)
    print("audio_energy:", val)