import numpy as np
import librosa
import warnings
from audioLIME.factorization_base import Factorization
import torch


def initialize_baseline(baseline_type, x):
    baseline = torch.zeros_like(x)
    if baseline_type == "min":
        baseline = baseline + x.min()
    elif baseline_type == "max":
        baseline = baseline + x.max()
    elif baseline_type == "mean":
        baseline = baseline + x.mean()
    elif baseline_type == "unif":
        baseline = torch.rand_like(x) * x.max()
    elif baseline_type == "shuffle":
        def _1d_to2d(n_col, i):
            # https://softwareengineering.stackexchange.com/a/212813/91332
            new_col = i % n_col
            new_row = i // n_col
            return new_col, new_row

        n_rows, n_cols = x.shape
        index = np.arange(n_rows * n_cols)
        np.random.shuffle(index)
        for orig_i, new_i in enumerate(index):
            new_col, new_row = _1d_to2d(n_cols, new_i)
            orig_col, orig_row = _1d_to2d(x.shape[1], orig_i)
            baseline[new_row, new_col] = x[orig_row, orig_col]
    return baseline


class TimeFrequencyTorchFactorization(Factorization):
    """
    Implements time-frequency segmentation as introduced by Mishra et al.
    to be used within audioLIME.

    Mishra, Saumitra, Bob L. Sturm, and Simon Dixon.
    "Local Interpretable Model-Agnostic Explanations for Music Content Analysis." ISMIR. 2017.

    """
    def __init__(self, input, target_sr, frequency_segments=4, temporal_segmentation_params=6, hop_length=256, baseline="zero",
                 composition_fn=None):
        assert len(input.shape) == 2
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn=composition_fn)
        assert baseline in ["zero", "min", "mean", "unif", "shuffle", "max"]
        self.n_frequency_segments = frequency_segments
        self.spectrogram = input
        self.hop_length=hop_length
        self.baseline_type = baseline
        self.baseline = initialize_baseline(baseline_type=baseline, x=self.spectrogram)


    def compose_model_input(self, components=None):
        return self._composition_fn(self.retrieve_components(components))

    def get_number_components(self):
        return self.n_frequency_segments * len(self.temporal_segments)

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.spectrogram

        if len(selection_order) > 0:
            max_val = max(selection_order)
            if max_val >=self.get_number_components():
                raise ValueError("{} out of bounds for {} components", max_val, self.get_number_components())

        mask = torch.zeros_like(self.spectrogram)
        unmask = torch.ones_like(self.spectrogram)

        # following the order of segments in [Mishra 2017] Figure 4
        temp_length = mask.shape[1] // len(self.temporal_segments)
        freq_length = mask.shape[0] // self.n_frequency_segments

        left_over = mask.shape[1] - temp_length * len(self.temporal_segments)
        if left_over > 0:
            warnings.warn("Adding last {} frames to last segment".format(left_over))

        def compute_f_start(f):
            return f * freq_length

        def compute_f_end(f):
            return compute_f_start(f) + freq_length

        for so in selection_order:
            t = so // self.n_frequency_segments  # index of temporal_segment
            # print("t", t)
            f = so % self.n_frequency_segments

            [t_start, t_end] = librosa.samples_to_frames(self.temporal_segments[t], hop_length=self.hop_length)
            if t == len(self.temporal_segments) - 1:
                t_end = mask.shape[1]
            # print("t_start {}, t_end{}".format(t_start, t_end))
            f_start = compute_f_start(f)
            f_end = compute_f_end(f)
            mask[f_start:f_end, t_start:t_end] = 1.
            unmask[f_start:f_end, t_start:t_end] = 0.

        return self.spectrogram * mask + self.baseline * unmask

