import unittest
import librosa
import numpy as np
from audioLIME.audio_utils import load_audio
from audioLIME.factorization_spleeter import SpleeterFactorization

class TestSpleeterFactorization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_sr = 16000
        self.audio_path = librosa.util.example_audio_file()
        self.audio = load_audio(self.audio_path, target_sr)
        self.reference, _ = librosa.load(self.audio_path, sr=target_sr)

    def test_SumAllComponents(self):
        factorization = SpleeterFactorization(self.audio, temporal_segmentation_params=1,
                                              composition_fn=None,
                                              model_name='spleeter:5stems')
        all_components = factorization.compose_model_input()
        self.assertTrue(np.allclose(all_components, self.reference, atol=10**5))

    def test_TemporalSegmentation(self):
        n_segments = 7
        factorization = SpleeterFactorization(self.audio, temporal_segmentation_params=n_segments,
                                              composition_fn=None,
                                              model_name='spleeter:5stems')
        all_components = factorization.compose_model_input()
        leng = len(all_components)  # to deal with ignored samples at the end
        self.assertTrue(np.allclose(all_components, self.reference[:leng], atol=10 ** 5))
        self.assertEqual(n_segments * 5, factorization.get_number_components()) # nr. sources = 5


if __name__ == '__main__':
    unittest.main()
