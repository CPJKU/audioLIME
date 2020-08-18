import numpy as np
import librosa
import warnings
from audioLIME.factorization_base import Factorization


class SoundLIMEFactorization(Factorization):
    """
    Implements time-frequency segmentation as introduced by Mishra et al.
    to be used with audioLIME.

    Mishra, Saumitra, Bob L. Sturm, and Simon Dixon.
    "Local Interpretable Model-Agnostic Explanations for Music Content Analysis." ISMIR. 2017.

    """
    def __init__(self, audio_path, frequency_segments=4, temporal_segments=6, sr=16000):
        super().__init__()
        # TODO: could also derive from DataBasedFactorization
        self.frequency_segments = frequency_segments
        self.temporal_segments = temporal_segments
        y, _ = librosa.load(audio_path, sr=sr)
        self.sr = sr
        self.original_mix = y

    def set_analysis_window(self, start_sample, y_length):
        self.mix = self.original_mix[start_sample:start_sample+y_length]

        D = librosa.stft(self.mix)
        temp_length = D.shape[1] // self.temporal_segments
        actual_length = temp_length * self.temporal_segments
        if actual_length < D.shape[1]:
            warnings.warn("Last {} frames are ignored".format(D.shape[1] - actual_length))
        D = D[:, :actual_length]

        mag, phase = librosa.magphase(D)
        self.phase = phase
        self.spectrogram = mag

        assert self.spectrogram.shape[0] % self.frequency_segments == 0, \
            "spec height {} must be a multiple of frequency_segments {} (for now)".format(mag.shape[0], self.frequency_segments)

    def compose_model_input(self, components=None):
        S = self.retrieve_components(components)
        D_ = S * self.phase
        y_ = librosa.istft(D_, length=len(self.mix))
        return y_

    def get_number_components(self):
        return self.frequency_segments * self.temporal_segments

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.spectrogram

        S = np.zeros_like(self.spectrogram)

        # following the order of segments in [Mishra 2017] Figure 4
        temp_length = S.shape[1] // self.temporal_segments
        freq_length = S.shape[0] // self.frequency_segments

        for so in selection_order:
            t = so // self.frequency_segments
            f = so % self.frequency_segments

            t_start = t * temp_length
            t_end = t_start + temp_length
            f_start =  f * freq_length
            f_end = f_start + freq_length

            S[f_start:f_end, t_start:t_end] = self.spectrogram[f_start:f_end, t_start:t_end]

        return S

