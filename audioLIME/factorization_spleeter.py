import numpy as np
import librosa
from audioLIME.factorization_base import SourceSeparationBasedFactorization
import os
import pickle

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    from spleeter.separator import Separator
except ImportError:
    Separator = None


class SpleeterFactorization(SourceSeparationBasedFactorization):
    # TODO: order of parameters is messed up (make same as in other classes)
    def __init__(self, input, temporal_segmentation_params, composition_fn, target_sr=16000,
                 model_name="spleeter:5stems"):
        self.model_name = model_name
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)

    def initialize_components(self):
        spleeter_sr = 44100

        waveform = self._original_mix
        separator = Separator(self.model_name, multiprocess=False)
        waveform = librosa.resample(waveform, self.target_sr, spleeter_sr)
        waveform = np.expand_dims(waveform, axis=1)
        prediction = separator.separate(waveform)

        original_components = [
            librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr) for
            key in prediction]

        components_names = list(prediction.keys())
        return original_components, components_names


def pickle_dump(x, path):
    pickle.dump(x, open(path, "wb"))


def pickle_load(path):
    return pickle.load(open(path, "rb"))


class SpleeterPrecomputedFactorization(SpleeterFactorization):
    def __init__(self, input, temporal_segmentation_params, composition_fn, target_sr=16000,
                 model_name="spleeter:5stems", spleeter_sources_path=None, recompute=False):
        assert isinstance(input, str), "input must be file path. otherwise use SpleeterFactorization."
        if spleeter_sources_path is None:
            raise TypeError("spleeter_sources_path must not be None. "
                            "Provide path or use SpleeterFactorization.")
        self.spleeter_sources_path = os.path.join(spleeter_sources_path, model_name)
        assert os.path.exists(self.spleeter_sources_path)
        self.recompute = recompute
        super().__init__(input, temporal_segmentation_params, composition_fn, target_sr, model_name)

    def initialize_components(self):
        spleeter_sr = 44100
        precomputed_name = os.path.basename(self._audio_path) + ".pt"
        precomputed_path = os.path.join(self.spleeter_sources_path, precomputed_name)
        if self.recompute:
            waveform = self._original_mix
            separator = Separator(self.model_name, multiprocess=False)
            waveform = librosa.resample(waveform, self.target_sr, spleeter_sr)
            waveform = np.expand_dims(waveform, axis=1)
            prediction = separator.separate(waveform)
            pickle_dump(prediction, precomputed_path)
        else:
            prediction = pickle_load(precomputed_path)

        original_components = [
            librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr) for
            key in prediction]

        components_names = list(prediction.keys())
        return original_components, components_names


if __name__ == '__main__':
    source_path = '/share/cp/datasets/svd_jamendo/audio/'
    destination_path = '/share/cp/projects/xaimir_benchmark/source_separation/spleeter/jamendo'
    audio_list = os.listdir(source_path)
    model_name = "spleeter:2stems"

    for i, audio in enumerate(audio_list):
        print("processing {}/{} {}".format(i+1, len(audio_list), audio))
        input = os.path.join(source_path, audio)
        factorization = SpleeterPrecomputedFactorization(input,
                                                         temporal_segmentation_params=1,
                                                         composition_fn=None,
                                                         spleeter_sources_path=destination_path,
                                                         recompute=True,
                                                         target_sr=44100,
                                                         model_name=model_name)
        print("mix length:", len(factorization._original_mix))
