import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as tor
from argparse import ArgumentParser




""" Parameters """
VIDEOS_TEST_FP = "./videos_valid_0.npy"
LABELS_TEST_FP = "./labels_valid_0.npy"




def plot_tsne(mode, model_fp, limit) :

    videos = np.load(VIDEOS_TEST_FP)
    labels = np.load(LABELS_TEST_FP)


    model = tor.load(model_fp).cuda()

    correct, total = 0, len(labels)

    features_rnn = []

    for i, (x, label) in enumerate(zip(videos, labels), 1) :
        print ("Process: {}/{}".format(i, total))
        x = tor.Tensor(x).unsqueeze(0).cuda()
        f = model.get_feature(x)
        features_rnn.append(f)

    print (features_rnn)




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, choices=["valid", "train"])
    parser.add_argument("-l", type=int, default=None, help="limitation of amount of data")
    parser.add_argument("--load", type=str, required=True, help="loaded model file path")

    mode = parser.parse_args().m
    model_fp = parser.parse_args().load
    limit = parser.parse_args().l

    plot_tsne(mode, model_fp, limit)