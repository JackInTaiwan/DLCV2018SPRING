import cv2
import torch as tor
import numpy as np

from argparse import ArgumentParser
from torch.autograd import Variable
from utils import normalize, select_data
from train import VIDEOS_MAX_BATCH




""" Parameters """
VIDEOS_TEST_FP = "./videos_valid_0.npy"
LABELS_TEST_FP = "./labels_valid_0.npy"




def evaluation(mode, model_fp, limit) :
    if mode == "test" :
        videos = np.load(VIDEOS_TEST_FP)
        labels = np.load(LABELS_TEST_FP)

    if limit :
        videos = videos[:limit]
        labels = labels[:limit]

    videos = normalize(videos / 255.)
    videos = select_data(videos, VIDEOS_MAX_BATCH)

    model = tor.load(model_fp).cuda()

    correct, total = 0, len(labels)

    for i, (x, label) in enumerate(zip(videos, labels), 1) :
        print ("Process: {}/{}".format(i, total))
        x = Variable(tor.FloatTensor(x)).permute(0, 3, 1, 2).cuda()
        out = model(x)
        out = out.mean(dim=0).unsqueeze(0)
        pred = model.pred(out)
        y = tor.max(pred, 1)[1]
        if int(y[0].data) == label :
            correct += 1

    acc = correct / total

    print ("|Acc on {}: {}".format(mode, round(acc, 6)))

    return acc




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, choices=["test", "train"])
    parser.add_argument("-l", type=int, default=None, help="limitation of amount of data")
    parser.add_argument("--model", type=str, required=True)

    mode = parser.parse_args().m
    model_fp = parser.parse_args().model
    limit = parser.parse_args().l

    evaluation(mode, model_fp, limit)
