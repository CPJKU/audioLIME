import numpy as np
import warnings
import os
import librosa
from audioLIME.factorization_base import Factorization
from audioLIME.data_provider import RawAudioProvider
import pickle

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from spleeter.separator import Separator
except ImportError:
    Separator = None


def default_composition_fn(x):
    return x


class DataBasedFactorization(Factorization):

    def __init__(self, data_provider, n_temporal_segments, composition_fn=None):
        """
        :param data_provider: object of class DataProvider
        :param n_temporal_segments: number of temporal segments used in the segmentation
        :param composition_fn: allows to apply transformations to the summed sources,
                e.g. return a spectrogram
                (same factorization class can be used independent of the input the model requires)
        """
        super().__init__()
        if composition_fn is None:
            composition_fn = default_composition_fn
        self.data_provider = data_provider
        self.composition_fn = composition_fn
        self.n_temporal_segments = n_temporal_segments
        self.original_components = []
        self.components = []
        self._components_names = []

        self.initialize_components()  # that's the part that's specific to each source sep. algorithm
        self.set_analysis_window(0, len(self.data_provider.get_mix()))

    def compose_model_input(self, components=None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = sum(sel_sources)
        else:
            y = sel_sources[0]
        return self.composition_fn(y)

    def get_number_components(self):
        return len(self.components)

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.components
        return [self.components[o] for o in selection_order]

    def get_ordered_component_names(self):
        if len(self._components_names) == 0:
            raise Exception("Components were not named.")
        return self._components_names

    def initialize_components(self):
        raise NotImplementedError

    def prepare_components(self, start_sample, y_length):
        # this resets in case temporal segmentation was previously applied
        self.components = [
            comp[start_sample:start_sample + y_length] for comp in self.original_components]

        mix = self.data_provider.get_mix()
        audio_length = len(mix)
        n_temporal_segments = self.n_temporal_segments
        samples_per_segment = audio_length // n_temporal_segments

        explained_length = samples_per_segment * n_temporal_segments
        if explained_length < audio_length:
            warnings.warn("last {} samples are ignored".format(audio_length - explained_length))

        component_names = []
        temporary_components = []
        for s in range(n_temporal_segments):
            segment_start = s * samples_per_segment
            segment_end = segment_start + samples_per_segment
            for co in range(self.get_number_components()):
                # current_component = np.zeros(explained_length, dtype=np.float32)
                current_component = torch.cuda.FloatTensor(explained_length) # TODO: make this variable!
                current_component[segment_start:segment_end] = self.components[co][segment_start:segment_end]
                temporary_components.append(current_component)
                component_names.append(self._components_names[co]+str(s))

        self.components = temporary_components
        self._components_names = component_names

    def set_analysis_window(self, start_sample, y_length):
        self.data_provider.set_analysis_window(start_sample, y_length)
        self.prepare_components(start_sample, y_length)


def separate(separator, waveform, target_sr, spleeter_sr):
    waveform = np.expand_dims(waveform, axis=0)
    waveform = librosa.resample(waveform, target_sr, spleeter_sr)
    waveform = np.swapaxes(waveform, 0, 1)
    prediction = separator.separate(waveform)
    return prediction

class SpleeterFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn, model_name, target_sr=16000):
        assert isinstance(data_provider, RawAudioProvider)  # TODO: nicer check
        self.model_name = model_name
        self.target_sr = target_sr
        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        spleeter_sr = 44100

        if Separator is None:
            raise ImportError('spleeter is not imported')

        separator = Separator(self.model_name, multiprocess=False)

        # Perform the separation:
        waveform = self.data_provider.get_mix()
        prediction = separate(separator, waveform, self.target_sr, spleeter_sr)

        self.original_components = [
            librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr) for
            key in prediction]
        self._components_names = list(prediction.keys())


class SpleeterPrecomputedFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn, model_name,
                 spleeter_sources_path, target_sr=16000):
        assert isinstance(data_provider, RawAudioProvider)  # TODO: nicer check
        self.model_name = model_name
        self.target_sr = target_sr
        sample_name = os.path.basename(data_provider.get_audio_path().replace(".mp3", ""))
        self.sources_path = os.path.join(spleeter_sources_path,
                                         model_name.replace("spleeter:", ""), sample_name)
        # print(self.sources_path)

        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        spleeter_sr = 44100

        prediction_path = os.path.join(self.sources_path, "prediction.pt")
        if not os.path.exists(prediction_path):
            separator = Separator(self.model_name, multiprocess=False)
            waveform = self.data_provider.get_mix()
            prediction = separate(separator, waveform, self.target_sr, spleeter_sr)
            pickle.dump(prediction, open(prediction_path, "wb"))
        else:
            print("Loading", prediction_path)

        prediction = pickle.load(open(prediction_path, "rb"))

        self.original_components = [
            torch.cuda.FloatTensor(
                librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr))
            for key in prediction]
        self._components_names = list(prediction.keys())

    def compose_model_input(self, components=None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = torch.stack(sel_sources, dim=0).sum(dim=0)
        else:
            y = sel_sources[0]
        return self.composition_fn(y)

