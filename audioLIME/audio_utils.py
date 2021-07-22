"""
Contains utils functions related to audio loading, ...
"""

import librosa


def load_audio(audio_path, target_sr):
    waveform, _ = librosa.load(audio_path, mono=True, sr=target_sr)
    return waveform
