import unittest
import librosa
import numpy as np
from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import SpleeterFactorization

class TestSpleeterFactorization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_path = librosa.util.example_audio_file()
        self.dp = RawAudioProvider(self.audio_path)
        self.reference, _ = librosa.load(self.audio_path, sr=16000)

    def test_SumAllComponents(self):
        factorization = SpleeterFactorization(self.dp, n_temporal_segments=1,
                                              composition_fn=None,
                                              model_name='spleeter:5stems')
        all_components = factorization.compose_model_input()
        self.assertTrue(np.allclose(all_components, self.reference, atol=10**5))

    def test_AnalysisWindow(self):
        start = 35000
        leng = 27333
        reference = self.reference[start:start+leng]
        factorization = SpleeterFactorization(self.dp, n_temporal_segments=1,
                                              composition_fn=None,
                                              model_name='spleeter:5stems')
        factorization.set_analysis_window(start, leng)
        all_components = factorization.compose_model_input()
        self.assertTrue(np.allclose(all_components, reference, atol=10 ** 5))

    def test_TemporalSegmentation(self):
        n_segments = 7
        factorization = SpleeterFactorization(self.dp, n_temporal_segments=n_segments,
                                              composition_fn=None,
                                              model_name='spleeter:5stems')
        all_components = factorization.compose_model_input()
        leng = len(all_components)  # to deal with ignored samples at the end
        self.assertTrue(np.allclose(all_components, self.reference[:leng], atol=10 ** 5))
        self.assertEqual(n_segments * 5, factorization.get_number_components()) # nr. sources = 5


if __name__ == '__main__':
    unittest.main()
