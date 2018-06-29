import os
import torch as tor
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import MODELS




def load_model(model_version, model_fp) :
    Model = MODELS[model_version - 1]
    model = Model()
    model.load_state_dict(state_dict=tor.load(model_fp))

    return model



def load_novel_class(shot) :
    novel_data_dp = "./task2-dataset/novel/"

    novel_support = np.zeros((20, shot, 32, 32, 3))

    for label_idx, dir_name in enumerate(sorted(os.listdir(novel_data_dp))):
        train_fp = os.path.join(novel_data_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))[:shot]):
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            img = (img - 0.5) * 2
            #img = img * 225.
            novel_support[label_idx][i] = img

    return novel_support



def evaluation(model, support_data, data_fp, output_fp) :
    model.cuda()
    model.eval()

    pred_list = []

    support_data = tor.Tensor(support_data).permute(0, 1, 4, 2, 3).cuda()
    pred_num = len(sorted(os.listdir(data_fp)))

    for i, fn in enumerate(sorted(os.listdir(data_fp))):
        print ("|Process: {}/{}".format(i + 1, pred_num), end="\r")
        img_fp = os.path.join(data_fp, fn)
        img = plt.imread(img_fp)
        img = (img - 0.5) * 2
        img = tor.Tensor(img).view(1, 32, 32, 3).permute(0, 3, 1, 2).cuda()
        pred = model(support_data, img)
        pred = tor.argmax(pred, dim=0).cpu()
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
    support_data = load_novel_class(shot)
    evaluation(model, support_data, data_fp, output_fp)

