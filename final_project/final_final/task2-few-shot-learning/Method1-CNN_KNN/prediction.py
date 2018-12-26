import os
import random
import torch as tor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import MODELS




def load_data(novel_dp, shot=5) :
    # novel loading
    # img shape = (32, 32, 3), pixel range=(0, 1)
    novel_support = np.zeros((20, shot, 32, 32, 3))
    novel_test = np.zeros((20, 500 - shot, 32, 32, 3))
    for label_idx, dir_name in enumerate(sorted(os.listdir(novel_dp))) :
        train_fp = os.path.join(novel_dp, dir_name, "train")
        fn_list = os.listdir(train_fp)
        random.shuffle(fn_list)
        for i, img_fn in enumerate(fn_list) :
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            img = (img - 0.5) * 2
            #img = img * 225.

            if i < shot:
                novel_support[label_idx][i] = img

            else:
                novel_test[label_idx][i - shot] = img

    return novel_support, novel_test



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
            #img = (img - 0.5) * 2
            img = img * 225.
            novel_support[label_idx][i] = img

    return novel_support



def evaluation(model, shot, support_data, data_fp, output_fp) :
    model.cuda()
    model.eval()
    table = [0, 10, 23, 30, 32, 35, 48, 54, 57, 59, 60, 64, 66, 69, 71, 82, 91, 92, 93, 95 ]

    pred_list = []

    support_data = tor.Tensor(support_data).permute(0, 1, 4, 2, 3).cuda()
    pred_num = len(sorted(os.listdir(data_fp)))

    for i in range(pred_num) :
        print ("|Process: {}/{}".format(i + 1, pred_num), end="\r")
        fn = "{}.png".format(i)
        img_fp = os.path.join(data_fp, fn)
        img = plt.imread(img_fp)
        img = (img - 0.5) * 2
        #img = img * 225.
        img = tor.Tensor(img).view(1, 32, 32, 3).permute(0, 3, 1, 2).cuda()
        pred = model.pred(support_data, img)
        pred_list.append([i, "{:0>2}".format(table[int(pred[0])])])

    pred_df = pd.DataFrame(pred_list)
    pred_df.to_csv(os.path.join(output_fp, "prediction_{}_shot.csv".format(shot)), header=["image_id", "predicted_label"], index=None)




if __name__ == "__main__" :
    """ Parameters """
    NOVEL_DIR_FP = "./task2-dataset/novel/"
    BASE_DIR_FP = "./task2-dataset/base/"
    RECORDS_FP = "./records/"

    WAY = 5

    parser = ArgumentParser()

    parser.add_argument("--way", type=int, default=20, help="number of way")
    parser.add_argument("--shot", type=int, default=5, help="number of shot")
    parser.add_argument("--load", type=str, required=True, help="the fp of model you want to load")
    parser.add_argument("--version", type=int, required=True, help="version of model")
    parser.add_argument("--data", type=str, required=True, help="data dir path")
    parser.add_argument("--output", type=str, required=True, help="file path for output" )
    parser.add_argument("--novel", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed")

    way = parser.parse_args().way
    shot = parser.parse_args().shot
    model_fp = parser.parse_args().load
    model_version = parser.parse_args().version
    data_fp = parser.parse_args().data
    novel_dir_fp = parser.parse_args().novel
    output_fp = parser.parse_args().output
    seed = parser.parse_args().seed

    random.seed(seed)

    novel_support, novel_test = load_data(novel_dir_fp, shot)
    model = load_model(model_version, model_fp)
    #support_data = load_novel_class(shot)
    evaluation(model, shot, novel_support, data_fp, output_fp)

