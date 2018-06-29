import os
import torch as tor
import matplotlib.pyplot as plt
from main import MODELS
from argparse import ArgumentParser

from models import MODELS




def load_model(model_version, model_fp) :
    Model = MODELS[model_version - 1]
    model = Model
    model.load_state_dict(tor.load(model_fp))

    return model



def evaluation(model, data_fp, output_fp) :
    model.cuda()
    model.eval()

    pred_list = []

    for fn in sorted(os.listdir(data_fp)):
        img_fp = os.path.join(data_fp, fn)
        img = plt.imread(img_fp)
        pred = model(img)
        pred = tor.argmax(pred, dim=1).cpu()
        print (pred)
        pred_list.append(int(pred))

    print (pred_list)
    print (len(pred_list))




if __name__ == "__main__" :
    parser = ArgumentParser()

    parser.add_argument("--way", type=int, default=20, help="number of way")
    parser.add_argument("--shot", type=int, default=5, help="number of shot")
    parser.add_argument("--load", type=str, required=True, help="the fp of model you want to load")
    parser.add_argument("--version", type=int, required=True, help="version of model")
    parser.add_argument("--data", type=str, required=True, help="data dir path")
    parser.add_argument("--output", type=str, required=True, help="file path for output" )

    way = parser.parse_args().way
    shot = parser.parse_args().shot
    model_fp = parser.parse_args().load
    model_version = parser.parse_args().version
    data_fp = parser.parse_args().data
    output_fp = parser.parse_args().output

    model = load_model(model_version, model_fp)

    evaluation(model, data_fp, output_fp)

