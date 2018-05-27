import os
import cv2
import time
import numpy as np
import torch as tor
import torchvision.datasets as datasets

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

try :
    from .reader import readShortVideo, getVideoList
    from .utils import normalize, console, Batch_generator
    from .model import Classifier
except :
    from reader import  readShortVideo, getVideoList
    from utils import normalize, console, Batch_generator
    from model import Classifier




""" Parameters """
TRIMMED_LABEL_TRAIN_PF = "./labels_train.npy"
TRIMMED_VIDEO_TRAIN_PF = "./videos_train.npy"

EPOCH = 30
BATCHSIZE = 1
LR = 0.0001

AVAILABLE_SIZE = None




""" Load Data """
def load() :
    videos = np.load(TRIMMED_VIDEO_TRAIN_PF) / 255.
    labels = np.load(TRIMMED_LABEL_TRAIN_PF)
    videos = normalize(videos)

    global AVAILABLE_SIZE
    AVAILABLE_SIZE = videos.shape[0]

    batch_gen = Batch_generator(
        x=videos,
        y=labels,
        batch=BATCHSIZE,
        drop_last=True,
    )

    return batch_gen




""" Training """
def train(batch_gen, model, model_index, x_eval_train) :
    epoch_start = model.epoch
    step_start = model.step

    optim = tor.optim.Adam(model.fc1.parameters(), lr=LR)
    loss_func = tor.nn.CrossEntropyLoss().cuda()

    for epoch in range(epoch_start, epoch_start + EPOCH) :
        print("|Epoch: {:>4} |".format(epoch))

        for step, (x_batch, y_batch) in enumerate(batch_gen, step_start):
            print("Process: {}/{}".format(step , int(AVAILABLE_SIZE / BATCHSIZE)), end="\r")

            x = Variable(tor.FloatTensor(x_batch[0])).permute(0, 3, 1, 2).cuda()
            y = Variable(tor.LongTensor(y_batch)).cuda()

            optim.zero_grad()
            out = model(x)
            cls = model.cls(out)

            loss = loss_func(cls, y)

            loss.backward()
            optim.step()




""" Main """
if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-e", action="store", type=int, help="epoch")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--load", action="store", type=str, help="file path of loaded model")

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i
    load_model_fp = parser.parse_args().load
    cpu = parser.parse_args().cpu
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    BATCHSIZE = parser.parse_args().bs if parser.parse_args().bs else BATCHSIZE
    EPOCH = parser.parse_args().e if parser.parse_args().e else EPOCH



    ### Data load
    console("Loading Data")
    batch_gen = load()


    ### Load Model
    console("Loading Model")
    if load_model_fp :
        pass
    else :
        model = Classifier()

    if not cpu :
        model.cuda()


    ### Train Data
    console("Training Data")
    x_eval_train = None
    train(batch_gen, model, model_index, x_eval_train)