import sys
import os
from argparse import Namespace
import torch
from torch.autograd import Variable
import numpy as np

path_sota = '/home/verena/deployment/sota-music-tagging-models/'

sys.path.append(path_sota)
sys.path.append(os.path.join(path_sota, 'training'))
from training.eval import Predict  # can only be imported after appending path_sota in sota_utils

tag_file = open(os.path.join(path_sota, "split", "msd", "50tagList.txt"), "r")
tags_msd = [t.replace('\n', '') for t in tag_file.readlines()]
path_models = os.path.join(path_sota, 'models')

# sota config
def prepare_config(model_type="fcn", input_length=29*16000):
    config = Namespace()
    config.dataset = "msd"  # we use the model trained on MSD
    config.model_type = model_type
    config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
    config.input_length = input_length
    config.batch_size = 1  # we analyze one chunk of the audio
    return config


def get_predict_fn(config):
    model = Predict.get_model(config)

    if torch.cuda.is_available():
        S = torch.load(config.model_load_path)
    else:
        S = torch.load(config.model_load_path, map_location="cpu")

    model.load_state_dict(S)
    model.cuda()
    model.eval()

    def predict_fn(x_array):
        # based on code from sota repo
        x = torch.zeros(len(x_array), config.input_length)
        for i in range(len(x_array)):
            x[i] = torch.Tensor(x_array[i]).unsqueeze(0)
        x = x.cuda()
        x = Variable(x)
        y = model(x)
        y = y.detach().cpu().numpy()
        return np.array(y)

    return predict_fn


def composition_fn(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x