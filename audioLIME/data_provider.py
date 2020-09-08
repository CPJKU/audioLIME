"""
Contains data providers used for all factorizations that inherit from DataBasedFactorization.
"""

import librosa


class DataProvider(object):
    """
    :class:`DataProvider` is the base class for creating a data provider that will be used
    by any factorization class derived from `DataBasedFactorization`.
    """
    def __init__(self, audio_path):
        self._audio_path = audio_path
        self._original_mix = self.initialize_mix()
        self._mix = self._original_mix

    def initialize_mix(self):
        raise NotImplementedError

    def get_mix(self):
        return self._mix

    def get_audio_path(self):
        return self._audio_path

    def set_analysis_window(self, start_sample, y_length):
        """
        :param start_sample: index of the sample where the analysis window starts
        :param y_length: length (in samples) of the analysis window
        """
        self._mix = self._original_mix[start_sample:start_sample+y_length]

class RawAudioProvider(DataProvider):
    """
    :class:`RawAudioProvider` is used when the factorization algorithm requires raw audio.
    """
    def __init__(self, audio_path, target_sr=16000):
        self.target_sr = target_sr
        super().__init__(audio_path)

    def initialize_mix(self):
        waveform, _ = librosa.load(self._audio_path, mono=True, sr=self.target_sr)
        return waveform
