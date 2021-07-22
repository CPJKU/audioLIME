from audioLIME.factorization_base import Factorization
import skimage.segmentation as segmentation
import numpy as np

# skimage: https://scikit-image.org/docs/dev/api/skimage.segmentation.html

image_segmentation_algorithm_options = ["slic", "fsz"]  # TODO: add more

class ImageLikeFactorization(Factorization):
    def __init__(self, input, target_sr, torchaudio_spec,
                 image_segmentation_algorithm="slic", image_segmentation_params=None,
                 baseline="zero", temporal_segmentation_params=None):
        # TODO: pass spectrogram
        assert image_segmentation_algorithm in image_segmentation_algorithm_options
        assert baseline in ["zero", "min"]
        assert len(torchaudio_spec.shape) == 2
        if temporal_segmentation_params is not None:
            raise ValueError("temporal_segmentation_params can not be used with ImageLikeFactorization")
        temporal_segmentation_params = {'type': 'fixed_length',
                                        'n_temporal_segments': 1}
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn=None)
        self.spectrogram = torchaudio_spec.detach().cpu().numpy()
        self.baseline = baseline
        if image_segmentation_params is None:
            if image_segmentation_algorithm == "slic":
                image_segmentation_params = {} # to disable warning
            elif image_segmentation_algorithm == "fsz":
                image_segmentation_params = {
                    "scale": 25,
                    "min_size": 40
                }
        if image_segmentation_algorithm == "slic":
            image_segmentation_params["start_label"] = 1 # to disable warning
        self.image_segmentation_algorithm = image_segmentation_algorithm
        self.image_segmentation_params = image_segmentation_params
        self.original_components, self._components_names = self.initialize_components()

    def initialize_components(self):
        algorithm = self.image_segmentation_algorithm
        params = self.image_segmentation_params
        segments = []
        if algorithm == "slic":
            segments = segmentation.slic(self.spectrogram, **params)
        elif algorithm == "fsz":
            # Finally, regarding possible image segmentation algorithms
            # (Felzenszwalb, SLIC, Chan Vese, Watershed),
            # experiments showed that Felzenszwalb (with scale=25 and minsize=40)
            # provided the most reasonable visual segmentation of the spectrograms.
            segments = segmentation.felzenszwalb(self.spectrogram, **params)
        unique_components = np.unique(segments)
        segment_names = ["S" + str(nr) for nr in unique_components]
        return segments, segment_names

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.spectrogram

        max_val = max(selection_order)
        if max_val >=self.get_number_components():
            raise ValueError("{} out of bounds for {} components", max_val, self.get_number_components())

        mask = np.zeros_like(self.spectrogram)
        unmask = np.ones_like(self.spectrogram)
        baseline = np.zeros_like(self.spectrogram)

        if self.baseline == "min":
            baseline = baseline + self.spectrogram.min()

        for so in selection_order:
            mask_idx = self.original_components == so + 1  # skimage starts counting at 1, lime at 0
            mask[mask_idx] = 1.
            unmask[mask_idx] = 0.

        return self.spectrogram * mask + baseline * unmask

