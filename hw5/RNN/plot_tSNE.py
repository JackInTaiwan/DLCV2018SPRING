import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as tor

from argparse import ArgumentParser
from sklearn.manifold.t_sne import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




""" Parameters """
VIDEOS_TEST_FP = "./videos_valid_0.npy"
LABELS_TEST_FP = "./labels_valid_0.npy"




def plot_tsne(model_fp, output_fp, limit) :

    videos = np.load(VIDEOS_TEST_FP)
    labels = np.load(LABELS_TEST_FP)

    model = tor.load(model_fp)
    model.cuda()

    correct, total = 0, len(labels)

    features_rnn = []
    labels = []

    for i, (x, label) in enumerate(zip(videos, labels), 1) :
        print ("Process: {}/{}".format(i, total))
        x = tor.Tensor(x).unsqueeze(0).cuda()
        f = model.get_feature(x)
        features_rnn.append(f)

        labels.append(int(label))


    ### tSNE
    tsne = TSNE(
        n_components=2,
        random_state=0,
    )
    f_tsne = tsne.fit_transform(features_rnn)

    plt.scatter(f_tsne[labels == 0, 0], f_tsne[labels == 0, 1], c="r")
    plt.scatter(f_tsne[labels == 1, 0], f_tsne[labels == 1, 1], c="b")
    #plt.legend(["Not {}".format(attr_selected), attr_selected])

    plt.savefig(os.path.join(output_fp, "tSNE_RNN.jpg"))




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-l", type=int, default=None, help="limitation of amount of data")
    parser.add_argument("--load", type=str, required=True, help="loaded model file path")
    parser.add_argument("--output", type=str, default=None)

    model_fp = parser.parse_args().load
    output_fp = parser.parse_args().output
    limit = parser.parse_args().l

    plot_tsne(model_fp, output_fp, limit)