import os
import cv2
import numpy as np
import torch as tor
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from butirecorder import Recorder
from train import Trainer

from models import (
    model_1,
)




def load_data(base_dp, novel_dp, shot=5) :
    # base_train loading
    base_train = np.empty((80, 500, 3, 32, 32))

    for label_idx, dir_name in enumerate(sorted(os.listdir(base_dp))) :
        train_fp = os.path.join(base_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))) :
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp).transpose(2, 0, 1)

            base_train[label_idx][i-shot] = img

    # novel loading
    # img shape = (32, 32, 3), pixel range=(0, 1)
    novel_support = np.empty((20, shot, 3, 32, 32))
    novel_test = np.empty((20, 500 - shot, 3, 32, 32))
    for label_idx, dir_name in enumerate(sorted(os.listdir(novel_dp))):
        train_fp = os.path.join(novel_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))):
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp).transpose(2, 0, 1)

            if i < shot:
                novel_support[label_idx][i] = img

            else:
                novel_test[label_idx][i - shot] = img

    print(base_train.shape, novel_support.shape, novel_test.shape)

    return base_train, novel_support, novel_test



def load_recorder(Model, model_index, record_dp, json_fn) :
    model = Model()
    recorder_name = "matchnet_{}".format(model_index)

    if json_fn == None :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            recorder_name=recorder_name,
            save_path=record_dp,
            models={
                "matchnet": model,
            }
        )

    else :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            models={
                "matchnet": model,
            }
        )
        recorder.load(json_fn)

    return recorder




if __name__ == "__main__" :
    """ Parameters """
    NOVEL_DIR_FP = "./task2-dataset/novel/"
    BASE_DIR_FP = "./task2-dataset/base/"
    RECORDS_FP = "./records/"

    MODELS = [model_1,]

    SHOT = 5
    LR = 0.0001
    EPOCH = 50


    """ Parser """
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=None, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=None, help="amount of validation data")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning rate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--load", action="store", type=str, default=None, help="the fn of json you want to load")
    parser.add_argument("--record", action="store", type=str, required=True, help="dir path of record")
    parser.add_argument("--version", action="store", type=int, default=0, help="version of model")

    limit = parser.parse_args().l
    valid_limit = parser.parse_args().v
    model_index = parser.parse_args().i
    load_model_fp = parser.parse_args().load
    cpu = parser.parse_args().cpu
    model_version = parser.parse_args().version
    record_dp = parser.parse_args().record
    json_fn = parser.parse_args().load
    LR = parser.parse_args().lr if parser.parse_args().lr else LR


    """ Main """
    base_train, novel_support, novel_test = load_data(BASE_DIR_FP, NOVEL_DIR_FP, shot=5)
    recorder = load_recorder(MODELS[model_version], model_index, record_dp, json_fn)

    trainer = Trainer(
        recorder=recorder,
        base_train=base_train,
        novel_support=novel_support,
        novel_test=novel_test,
        shot=SHOT,
        cpu=cpu,
    )

    trainer.train()

