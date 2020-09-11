class MadmomAudioProvider(DataProvider):
    def __init__(self, audio_path, target_sr=16000):
        from madmom.audio.signal import SignalProcessor
        self.target_sr = target_sr
        self.processor = SignalProcessor(num_channels=1, sample_rate=self.target_sr)
        super().__init__(audio_path)

    def initialize_mix(self):
        return self.processor(self._audio_path)
