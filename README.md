# audioLIME

![](imgs/audiolime.png)

This repository contains the Python package audioLIME, a tool for creating listenable explanations
for machine learning models in music information retrival (MIR).
audioLIME is based on the method lime (local interpretable model-agnostic explanations) work 
presented in [this paper](https://arxiv.org/abs/1602.04938) and uses source separation estimates in
order to create interpretable components.
If you use audioLIME in your work, please
use this [bib file for citation](citation.bib).

## Installation

The audioLIME package is not on PyPI yet. For installing it, clone the git repo and install it using 
`setup.py`.

```sh
git clone https://github.com/CPJKU/audioLIME.git  # HTTPS
```

```sh
git clone git@github.com:CPJKU/audioLIME.git  # SSH
```

```sh
cd audioLIME
python setup.py develop
```

or 

```sh
cd audioLIME
python setup.py install
```

If you do not know the difference between `develop` and `install` check 
[this article](http://naoko.github.io/your-project-install-pip-setup/).

### Note on Requirements

To keep it lightweight, not all possible dependencies are contained in `setup.py`. 
Depending on the factorization you want to use, you might need different packages, 
e.g. `nussl` or `spleeter`. 

### Tests

To test your installation, the following test are available:

`MODEL_PATH=/share/home/verena/experiments/spleeter/pretrained_models/  python -m unittest tests.test_SpleeterFactorization`

## Available Factorizations

Currently we have the following factorizations implemented:

* SpleeterFactorization based on the source separation system spleeter 
([code](https://github.com/deezer/spleeter/))
* CerberusFactorization based on Cerberus [arxiv](https://arxiv.org/abs/1910.12621)
* OVAFactorization based on OVA-models as described in the slakh paper 
[arxiv](https://arxiv.org/abs/1909.08494)
* SoundLIMEFactorization: time-frequency segmentation based on 
[SoundLIME](https://github.com/saum25/SoundLIME) 
(the original implementation was not flexible enough for our experiments)

## Usage Example

Here we demonstrate how we can explain the prediction of 
FCN ([code](https://github.com/minzwon/sota-music-tagging-models), 
[Choi 2016](https://arxiv.org/abs/1606.00298), 
[Won 2020](https://arxiv.org/abs/2006.00751)) using `SpleeterFactorization`.

For this to work you need to install the requirements found in the above mentioned repo of 
the tagger and spleeter:
```sh
pip install spleeter
```

```python
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
```

For the details on setting everything up, see 
[example_using_spleeter_fcn](examples/example_using_spleeter_fcn.py).

Listen to the [input](https://soundcloud.com/veroamilbe/hop-along-sister-input) 
and [explanation](https://soundcloud.com/veroamilbe/hop-along-sister-explanation).

## TODOs

* [ ] upload to [pypi.org](https://pypi.org) (to allow installation via `pip`)
* [ ] usage example for `SoundLIMEFactorization`