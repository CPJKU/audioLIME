from audioLIME.audio_utils import load_audio
import warnings
import numpy as np


def default_composition_fn(x):
    return x


def compute_segments(signal, sr, temporal_segmentation_params=None):
    # TODO: parameter for return type (samples, frames, seconds)?
    audio_length = len(signal)
    explained_length = audio_length
    if temporal_segmentation_params is None:
        n_temporal_segments_default = min(audio_length // sr, 10) # 1 segment per second, but maximally 10 segments
        temporal_segmentation_params = {'type': 'fixed_length',
                                        'n_temporal_segments': n_temporal_segments_default}
    elif isinstance(temporal_segmentation_params, int):
        temporal_segmentation_params = {'type': 'fixed_length',
                                        'n_temporal_segments': temporal_segmentation_params}

    segmentation_type = temporal_segmentation_params['type']
    assert segmentation_type in ['fixed_length', 'manual']

    segments = []
    if segmentation_type == "fixed_length":
        n_temporal_segments = temporal_segmentation_params['n_temporal_segments']
        samples_per_segment = audio_length // n_temporal_segments

        explained_length = samples_per_segment * n_temporal_segments
        if explained_length < audio_length:
            warnings.warn("last {} samples are ignored".format(audio_length - explained_length))

        for s in range(n_temporal_segments):
            segment_start = s * samples_per_segment
            segment_end = segment_start + samples_per_segment
            segments.append((segment_start, segment_end))
    elif segmentation_type == "manual":
        segments = temporal_segmentation_params["manual_segments"]
        explained_length = segments[-1][1]  # end of last segment

    return segments, explained_length


class Factorization(object):
    def __init__(self, input, target_sr, temporal_segmentation_params=None, composition_fn=None):
        self._audio_path = None
        self.target_sr = target_sr
        if isinstance(input, str):
            self._audio_path = input
            input = load_audio(input, target_sr)
        self._original_mix = input
        if composition_fn is None:
            composition_fn = default_composition_fn
        self._composition_fn = composition_fn

        self.original_components = []
        self.components = []
        self._components_names = []
        self.temporal_segments, self.explained_length = compute_segments(self._original_mix,
                                                                         self.target_sr,
                                                                         temporal_segmentation_params)

    def compose_model_input(self, components=None):
        return self._composition_fn(self.retrieve_components(components))

    def get_number_components(self):
        # TODO: probably no need to overwrite in other classes
        return len(self._components_names)

    def retrieve_components(self, selection_order=None):
        raise NotImplementedError

    def get_ordered_component_names(self): # e.g. instrument names
        return self._components_names


class TimeOnlyFactorization(Factorization):
    # TODO: add other baseline except 0's?
    def __init__(self, input, target_sr, temporal_segmentation_params=None, composition_fn=None):
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)
        for i in range(len(self.temporal_segments)):
            self._components_names.append("T"+str(i+1))

    def retrieve_components(self, selection_order=None):
        # TODO: check if selection_order contains out of bounds segments
        if selection_order is None:
            return self._original_mix
        retrieved_mix = np.zeros_like(self._original_mix)
        for so in selection_order:
            s, e = self.temporal_segments[so]
            retrieved_mix[s:e] = self._original_mix[s:e]
        return retrieved_mix


class SourceSeparationBasedFactorization(Factorization):

    def __init__(self, input, target_sr=16000, temporal_segmentation_params=None, composition_fn=None):
        """
        :param input: file_name of audio or numpy array containing waveform
        :param n_temporal_segments: number of temporal segments used in the segmentation
        :param composition_fn: allows to apply transformations to the summed sources,
                e.g. return a spectrogram
                (same factorization class can be used independent of the input the model requires)
        """
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)
        # the following part is specific to each source sep. algorithm
        self.original_components, self._components_names = self.initialize_components()
        self.prepare_components(0, len(self._original_mix))

    def compose_model_input(self, components=None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = sum(sel_sources)
        else:
            y = sel_sources[0]
        return self._composition_fn(y)

    def get_number_components(self):
        return len(self.components)

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.components
        if len(selection_order) == 0:
            return [np.zeros_like(self.components[0])]
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

        component_names = []
        temporary_components = []
        for s, (segment_start, segment_end) in enumerate(self.temporal_segments):
            for co in range(self.get_number_components()):
                current_component = np.zeros(self.explained_length, dtype=np.float32)
                current_component[segment_start:segment_end] = self.components[co][segment_start:segment_end]
                temporary_components.append(current_component)
                component_names.append(self._components_names[co] + str(s))

        self.components = temporary_components
        self._components_names = component_names


