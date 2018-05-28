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
    from .utils import normalize, select_data, console, accuracy, Batch_generator
    from .model import Classifier
except :
    from reader import  readShortVideo, getVideoList
    from utils import normalize, select_data, console, accuracy, Batch_generator
    from model import Classifier




""" Parameters """
TRIMMED_LABEL_TRAIN_FP = ["./labels_train_0.npy", "./labels_train_1.npy", "./labels_train_2.npy", "./labels_train_3.npy"]
TRIMMED_VIDEO_TRAIN_FP = ["./videos_train_0.npy", "./videos_train_1.npy", "./videos_train_2.npy", "./videos_train_3.npy"]

TRIMMED_LABEL_VALID_FP = "./labels_valid_0.npy"
TRIMMED_VIDEO_VALID_FP = "./videos_valid_0.npy"

EPOCH = 30
BATCHSIZE = 1
LR = 0.0001

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
VIDEOS_MAX_BATCH = 30
CAL_ACC_PERIOD = 300  # steps


""" Load Data """
def load(videos_fp, labels_fp, limit, val_limit) :
    videos = np.load(videos_fp)
    labels = np.load(labels_fp)
    videos = videos / 255.
    videos = normalize(videos)
    videos = select_data(videos, VIDEOS_MAX_BATCH)

    if limit :
        videos = videos[:limit]
        labels = labels[:limit]

    if val_limit :
        videos_eval = videos[:val_limit][:]
        labels_eval = labels[:val_limit][:]

    else :
        videos_eval = videos[:EVAL_TRAIN_SIZE][:]
        labels_eval = labels[:EVAL_TRAIN_SIZE][:]

    videos_test = normalize(np.load(TRIMMED_VIDEO_VALID_FP) / 255.)
    videos_test = select_data(videos_test, VIDEOS_MAX_BATCH)
    labels_test = np.load(TRIMMED_LABEL_VALID_FP)

    global AVAILABLE_SIZE
    AVAILABLE_SIZE = videos.shape[0]

    batch_gen = Batch_generator(
        x=videos,
        y=labels,
        batch=BATCHSIZE,
        drop_last=True,
    )

    return batch_gen, videos_eval, labels_eval, videos_test, labels_test




""" Training """
def train(model, model_index, limit, valid_limit) :
    epoch_start = model.epoch
    step_start = model.step

    optim = tor.optim.Adam(model.fc_1.parameters(), lr=LR)
    #optim = tor.optim.SGD(model.fc_1.parameters(), lr=LR)
    optim_vgg = tor.optim.Adam(model.vgg16.parameters(), lr=LR)
    loss_func = tor.nn.CrossEntropyLoss().cuda()

    loss_total = np.array([])

    for epoch in range(epoch_start, epoch_start + EPOCH) :
        print("|Epoch: {:>4} |".format(epoch))

        for videos_fp, labels_fp in zip(TRIMMED_VIDEO_TRAIN_FP, TRIMMED_LABEL_TRAIN_FP) :
            batch_gen, x_eval_train, y_eval_train, x_eval_test, y_eval_test = load(videos_fp, labels_fp, limit, valid_limit)

            for step, (x_batch, y_batch) in enumerate(batch_gen, step_start):
                print("Process: {}/{}".format(step , int(AVAILABLE_SIZE / BATCHSIZE)))
                x = Variable(tor.FloatTensor(x_batch[0])).permute(0, 3, 1, 2).cuda()
                y = Variable(tor.LongTensor(y_batch)).cuda()

                optim.zero_grad()
                optim_vgg.zero_grad()

                out = model(x)
                out = out.mean(dim=0).unsqueeze(0)
                pred = model.pred(out)

                loss = loss_func(pred, y)
                loss_total = np.concatenate((loss_total, [loss.data.cpu().numpy()]))
                #print ("|Loss: {}".format(loss.data.cpu().numpy()))
                loss.backward()

                optim.step()
                optim_vgg.step()

                model.step()

                if step % 20 == 0 :
                    print (loss_total.mean())
                    loss_total = np.array([])

                if step % CAL_ACC_PERIOD == 0 :
                    acc_train = accuracy(model, x_eval_train, y_eval_train)
                    acc_test = accuracy(model, x_eval_test, y_eval_test)

                    print ("|Acc on train data: {}".format(round(acc_train, 5)))
                    print ("|Acc on test data: {}".format(round(acc_test, 5)))

        model.epoch()



""" Main """
if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=None, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=None, help="amount of validation data")
    parser.add_argument("-e", action="store", type=int, help="epoch")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--load", action="store", type=str, help="file path of loaded model")

    limit = parser.parse_args().l
    valid_limit = parser.parse_args().v
    model_index = parser.parse_args().i
    load_model_fp = parser.parse_args().load
    cpu = parser.parse_args().cpu
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    BATCHSIZE = parser.parse_args().bs if parser.parse_args().bs else BATCHSIZE
    EPOCH = parser.parse_args().e if parser.parse_args().e else EPOCH


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
    train(model, model_index, limit, valid_limit)
