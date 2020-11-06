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
    def __init__(self, audio_path, frequency_segments=4, temporal_segments=6, sr=16000, mel_scale=False):
        super().__init__()
        # TODO: could also derive from DataBasedFactorization
        self.frequency_segments = frequency_segments
        self.temporal_segments = temporal_segments
        y, _ = librosa.load(audio_path, sr=sr)
        self.sr = sr
        self.original_mix = y
        self.mel_scale = mel_scale

    def set_analysis_window(self, start_sample, y_length):
        self.mix = self.original_mix[start_sample:start_sample+y_length]

        D = librosa.stft(self.mix)

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

        S = np.zeros_like(self.spectrogram) + self.spectrogram.min()

        # following the order of segments in [Mishra 2017] Figure 4
        temp_length = S.shape[1] // self.temporal_segments
        freq_length = S.shape[0] // self.frequency_segments

        left_over = S.shape[1] - temp_length * self.temporal_segments
        if left_over > 0:
            warnings.warn("Adding last {} frames to last segment".format(left_over))

        def compute_f_start(f):
            return f * freq_length

        def compute_f_end(f):
            return compute_f_start(f) + freq_length

        if self.mel_scale:
            f_max = self.sr // 2
            mel_max = librosa.hz_to_mel(f_max)
            hz_steps = librosa.mel_to_hz(list(range(0,
                                                    int(np.ceil(mel_max)),
                                                    int(mel_max // self.frequency_segments))))
            hz_steps[-1:] = f_max

            def compute_f_start(f):
                return int(hz_steps[f] / f_max * 1025)  # TODO don't hardcode this

            def compute_f_end(f):
                return int(hz_steps[f + 1] / f_max * 1025)

        for so in selection_order:
            t = so // self.frequency_segments
            f = so % self.frequency_segments

            t_start = t * temp_length
            if t == self.temporal_segments:
                t_end = S.shape[1]
            else:
                t_end = t_start + temp_length
            f_start = compute_f_start(f)
            f_end = compute_f_end(f)
            # print("f", f, f_start, f_end)

            S[f_start:f_end, t_start:t_end] = self.spectrogram[f_start:f_end, t_start:t_end]

        return S

