import os
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from butirecorder import Recorder
from train import Trainer

from models import (
    model_1,
    model_2,
    model_3,
    model_4,
    model_5,
    model_6,
    model_7,
    model_8,
    model_9,
    model_10,
    model_11,
    model_12,
    model_13,
    model_14,
    model_15,
    model_16,
    model_17,
    model_18,
    model_19,
)




def load_data(base_dp, novel_dp, shot=5) :
    # base_train loading
    base_train = np.zeros((80, 500, 32, 32, 3))

    for label_idx, dir_name in enumerate(sorted(os.listdir(base_dp))) :
        train_fp = os.path.join(base_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))) :
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            #img = (img - 0.5) * 2
            img = img * 225.

            base_train[label_idx][i-shot] = img

    # novel loading
    # img shape = (32, 32, 3), pixel range=(0, 1)
    novel_support = np.zeros((20, shot, 32, 32, 3))
    novel_test = np.zeros((20, 500 - shot, 32, 32, 3))
    for label_idx, dir_name in enumerate(sorted(os.listdir(novel_dp))):
        train_fp = os.path.join(novel_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))):
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            #img = (img - 0.5) * 2
            img = img * 225.

            if i < shot:
                novel_support[label_idx][i] = img

            else:
                novel_test[label_idx][i - shot] = img

    print(base_train.shape, novel_support.shape, novel_test.shape)

    return base_train, novel_support, novel_test



def load_recorder(Model, model_index, record_dp, json_fn, init) :
    model = Model()
    if init :
        model.init_weight()
    recorder_name = "relationnet_{}".format(model_index)

    if json_fn == None :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            recorder_name=recorder_name,
            save_path=record_dp,
            models={
                "relationnet": model,
            }
        )

    else :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            save_path=record_dp,
            models={
                "relationnet": model,
            }
        )
        recorder.load(json_fn)

    return recorder




if __name__ == "__main__" :
    """ Parameters """
    NOVEL_DIR_FP = "./task2-dataset/novel/"
    BASE_DIR_FP = "./task2-dataset/base/"
    RECORDS_FP = "./records/"

    MODELS = [
        model_1,
        model_2,
        model_3,
        model_4,
        model_5,
        model_6,
        model_7,
        model_8,
        model_9,
        model_10,
        model_11,
        model_12,
        model_13,
        model_14,
        model_15,
        model_16,
        model_17,
        model_18,
        model_19,
    ]

    WAY = 5
    SHOT = 5
    LR = 0.0001
    EPOCH = 50


    """ Parser """
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=None, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=None, help="amount of validation data")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--init", action="store_true", default=False, help="init weights of model")
    parser.add_argument("--step", type=int, default=None, help="limitation of steps for training")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning rate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--way", action="store", type=int, default=None, help="number of way")
    parser.add_argument("--shot", action="store", type=int, default=None, help="number of shot")
    parser.add_argument("--load", action="store", type=str, default=None, help="the fn of json you want to load")
    parser.add_argument("--record", action="store", type=str, required=True, help="dir path of record")
    parser.add_argument("--version", action="store", type=int, default=0, help="version of model")

    limit = parser.parse_args().l
    valid_limit = parser.parse_args().v
    model_index = parser.parse_args().i
    cpu = parser.parse_args().cpu
    init = parser.parse_args().init
    step = parser.parse_args().step
    model_version = parser.parse_args().version
    record_dp = parser.parse_args().record
    json_fn = parser.parse_args().load
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    WAY = parser.parse_args().way if parser.parse_args().way else WAY
    SHOT = parser.parse_args().shot if parser.parse_args().shot else SHOT


    """ Main """
    base_train, novel_support, novel_test = load_data(BASE_DIR_FP, NOVEL_DIR_FP, shot=5)
    recorder = load_recorder(MODELS[model_version - 1], model_index, record_dp, json_fn, init)

    trainer = Trainer(
        recorder=recorder,
        base_train=base_train,
        novel_support=novel_support,
        novel_test=novel_test,
        way=WAY,
        shot=SHOT,
        cpu=cpu,
        lr=LR,
        step=step,
    )

    trainer.train()


