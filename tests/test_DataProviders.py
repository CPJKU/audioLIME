import unittest
from audioLIME.data_provider import DataProvider, RawAudioProvider
import tempfile
import soundfile as sf
import numpy as np


class DummyDataProvider(DataProvider):
    def __init__(self, audio_path):
        super().__init__(audio_path)

    def initialize_mix(self):
        return None


class TestDataProviders(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sr = 16000
        self.temp_signal = np.random.randn(self.sr * 3) # create "fake" signal with 3 seconds length
        self.tmpfile = tempfile.NamedTemporaryFile()
        self.audio_path = self.tmpfile.name
        sf.write(self.audio_path, self.temp_signal, self.sr, format="wav")
        self.decimal_places = 5

    def test_BaseDataProvider(self):
        self.assertRaises(NotImplementedError, DataProvider, self.audio_path)

    def test_AudioPath(self):
        dp = DummyDataProvider(self.audio_path)
        self.assertEqual(self.audio_path, dp.get_audio_path())

    def test_RawAudioProviderMix(self):
        dp = RawAudioProvider(self.audio_path)
        self.assertTrue(np.allclose(dp.get_mix(), self.temp_signal, atol=10**self.decimal_places))

    def test_RawAudioAnalysisWindow(self):
        dp = RawAudioProvider(self.audio_path)
        reference_signal = dp.get_mix()
        start = 32000
        leng = 16000
        dp.set_analysis_window(start, leng)
        mix = dp.get_mix()
        self.assertAlmostEqual(mix[0], reference_signal[start], places=self.decimal_places)
        self.assertAlmostEqual(mix[-1], reference_signal[-1], places=self.decimal_places)
        self.assertEqual(len(mix), leng)
        self.assertTrue(np.allclose(dp._original_mix, self.temp_signal, atol=10**self.decimal_places))


if __name__ == '__main__':
    unittest.main()
