from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import SpleeterFactorization
from audioLIME import lime_audio
import soundfile as sf
import os

from examples.sota_utils import prepare_config, get_predict_fn

if __name__ == '__main__':
    audio_path = '/share/home/verena/data/samples/3_Hop Along-SisterCities.stem.mp4_sn0_original.wav'
    path_sota = '/home/verena/deployment/sota-music-tagging-models/'
    config = prepare_config("fcn", 29 * 16000)
    predict_fn = get_predict_fn(config)

    data_provider = RawAudioProvider(audio_path)
    spleeter_factorization = SpleeterFactorization(data_provider,
                                                   n_temporal_segments=10,
                                                   composition_fn=None,
                                                   model_name='spleeter:5stems')

    explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

    explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                             predict_fn=predict_fn,
                                             top_labels=1,
                                             num_samples=16384,
                                             batch_size=32
                                             )

    label = list(explanation.local_exp.keys())[0]
    top_components, component_indeces = explanation.get_sorted_components(label,
                                                                          positive_components=True,
                                                                          negative_components=False,
                                                                          num_components=3,
                                                                          return_indeces=True)

    print("predicted label:", label)
    sf.write(os.path.join("output", "explanation.wav"), sum(top_components), 16000)
    sf.write(os.path.join("output", "original.wav"),
             spleeter_factorization.data_provider.get_mix(), 16000)
