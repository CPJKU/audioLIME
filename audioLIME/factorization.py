import numpy as np
import warnings
import os
import librosa
from audioLIME.factorization_base import Factorization
from audioLIME.data_provider import RawAudioProvider, NusslAudioProvider

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

try:
    from nussl import DeepTranscriptionEstimation, DeepMaskEstimation
except ImportError:
    warnings.warn("Couldn't import stuff from nussl.")
    DeepTranscriptionEstimation = None
    AudioSignal = None


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
                current_component = np.zeros(explained_length, dtype=np.float32)
                current_component[segment_start:segment_end] = self.components[co][segment_start:segment_end]
                temporary_components.append(current_component)
                component_names.append(self._components_names[co]+str(s))

        self.components = temporary_components
        self._components_names = component_names

    def set_analysis_window(self, start_sample, y_length):
        self.data_provider.set_analysis_window(start_sample, y_length)
        self.prepare_components(start_sample, y_length)


class CerberusFactorization(DataBasedFactorization):

    def __init__(self, data_provider, n_temporal_segments, composition_fn, models_dir, model_name):
        assert isinstance(data_provider, NusslAudioProvider)  # TODO: nicer check
        self.models = {
            # Piano / Guitar Models
            'Cerberus - pno/gtr': '4335d73889e440b4bb41b4ed08f1ac4b',
            'Transcription only': '0d111525ad1a4fcd83a457b3107bcff9',
            'Transcription + Deep Clustering': 'ac5d47d989a946148887bca44c610568',
            'Chimera': '9bb99581c67e4aa2883d1d8f92622e25',
            'Mask Inference + Transcription': 'ceab2cbf61e04ef7b8f76d94a22c04d2',
            'Mask Inference': '72e51fa1b9c14393a4206171a9191f24',
            'Deep Clustering': '60f1e06a96a74ce7bc244f74f2f70eac',

            # Multi-instrument models
            'Cerberus - pno/gtr/bss': 'e7824712ddb4463b8fe66886eeeabc35',
            'Cerberus - pno/gtr/bss/drm': '3bc9840369804507adf6766ac400a116',
            'Cerberus - pno/gtr/bss/drm/str': 'b22be96d3ef5466d890ca8b11eb29544'
        }
        self.models_dir = models_dir
        self.models_name = model_name

        super().__init__(data_provider, n_temporal_segments, composition_fn)
        if DeepTranscriptionEstimation is None or AudioSignal is None:
            raise ImportError("DeepTranscriptionEstimation and/or AudioSignal from nussl are not imported.")

    def initialize_components(self):
        model_path = self.get_model_path(self.models_dir, self.model_name)
        mix = self.data_provider.get_mix().audio_data.squeeze()
        src_estimates = self.separate_mix(model_path)
        self.original_components = [src_estimates[src]['src_est'].audio_data.squeeze() for src
                           in src_estimates]
        sum_sources = sum(self.original_components)

        rest = mix - sum_sources
        self.original_components.append(rest)
        self._components_names = self.model_name.replace("Cerberus - ", "").split("/")
        self._components_names.append("rest")

    def separate_mix(self, model_dir):
        model_path = f'{model_dir}/best.model.pth'
        separator = DeepTranscriptionEstimation(
            self.data_provider.get_mix(),
            model_path=model_path
        )
        separator.run()

        result = {}
        estimates = separator.make_audio_signals()

        for i, est in enumerate(estimates):
            result[i] = {'src_est': est}

        return result

    def get_model_path(self, models_dir, model_name):
        return os.path.join(models_dir, self.models[model_name])


class OVAFactorization(DataBasedFactorization):

    def __init__(self, data_provider, n_temporal_segments, composition_fn, models_dir, add_rest=False):
        assert isinstance(data_provider, NusslAudioProvider)  # TODO: nicer check
        if DeepTranscriptionEstimation is None or AudioSignal is None:
            raise ImportError("DeepTranscriptionEstimation and/or AudioSignal from nussl are not imported.")

        self.add_rest = add_rest
        self.models = {
            'bass': os.path.join(models_dir, 'cookiecutter', 'slakh-bass', '8bfdadf6caf34888b2fe283e88b4320d'),
            'drums': os.path.join(models_dir, 'cookiecutter', 'slakh-drums', '8d59ac7a5f3142d08636a227b3646785'),
            'guitar': os.path.join(models_dir, 'cookiecutter', 'slakh-guitar', 'e8c21717979e4fcd8e5ba2b49bcd9310'),
            'piano': os.path.join(models_dir, 'cookiecutter', 'slakh-piano', '28920c91fe6242d89c94636de18577ac'),
        }

        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        for key in self.models:
            model_dir = self.models[key]
            self.original_components.append(self.separate_mix(model_dir).audio_data.squeeze())
            self._components_names.append(key)

        mix = self.data_provider.get_mix()
        mix = mix.audio_data.squeeze()
        if self.add_rest:
            sum_sources = sum(self.original_components)
            rest = mix - sum_sources
            self.original_components.append(rest)
            self._components_names.append("rest")

    def separate_mix(self, model_dir):
        checkpoint = os.path.join(model_dir, 'checkpoints', 'best.model.pth')
        model = DeepMaskEstimation(self.data_provider.get_mix(), checkpoint)
        model.run()
        estimates = model.make_audio_signals()
        return estimates[0]


class SpleeterFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn, model_name, target_sr=16000):
        assert isinstance(data_provider, RawAudioProvider)  # TODO: nicer check
        self.model_name = model_name
        self.target_sr = target_sr
        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        spleeter_sr = 41000

        if Separator is None:
            raise ImportError('spleeter is not imported')

        separator = Separator(self.model_name, multiprocess=False)

        # Perform the separation:
        waveform = self.data_provider.get_mix()
        waveform = np.expand_dims(waveform, axis=0)
        waveform = librosa.resample(waveform, self.target_sr, spleeter_sr)
        waveform = np.swapaxes(waveform, 0, 1)
        prediction = separator.separate(waveform)

        self.original_components = [
            librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr) for
            key in prediction]
        self._components_names = list(prediction.keys())
