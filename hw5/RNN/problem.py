import cv2
import os
import math
import torch as tor
import numpy as np

from argparse import ArgumentParser
from reader import getVideoList, readShortVideo
from utils import normalize, select_data
from train import VIDEOS_MAX_BATCH
from torchvision.transforms import Normalize




MAX_VIDEO_LEN = 10


def prediction(model_fp, vgg_fp, data_fp, label_fp, output_fp, limit) :
    l = getVideoList(label_fp)
    videos_output, labels_output = [], []

    data_num = limit if limit != None else len(l["Video_category"])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = Normalize(mean, std)

    vgg = tor.load(vgg_fp)
    vgg.cuda()

    for batch in range(data_num):
        print ("Convert videos into numpy: {}/{} \r".format(batch + 1, data_num), end="")

        cat = l["Video_category"][batch]
        name = l["Video_name"][batch]
        label = l["Action_labels"][batch]
        data = readShortVideo(data_fp, cat, name, downsample_factor=12)


        if len(data) > MAX_VIDEO_LEN :
            seq = [math.floor(data.shape[0] * _i / MAX_VIDEO_LEN) for _i in range(MAX_VIDEO_LEN)]
            data = data[seq]

        data = tor.Tensor(data).permute(0, 3, 1, 2) / 255.


        for i in range(len(data)) :
            data[i] = norm(data[i])
        data = data.cuda()
        out = vgg(data)
        features = out.cpu().data.numpy()
        videos_output.append(features)
        labels_output.append(int(label))

    vgg.cpu()

    features, labels = np.array(videos_output), np.array(labels_output)


    ### Prediction
    model = tor.load(model_fp)
    model.cuda()

    correct, total = 0, len(labels)

    for i, (x, label) in enumerate(zip(features, labels), 1) :
        print ("Process: {}/{}".format(i, total))
        x = tor.Tensor(x).unsqueeze(0).cuda()
        pred = model(x)
        pred = tor.max(pred, 1)[1]

        if int(pred[0].data) == label :
            correct += 1

    acc = correct / total

    print ("|Acc: {}".format(round(acc, 6)))

    return acc




if __name__ == "__main__" :
    parse = ArgumentParser()
    parse.add_argument("-l", type=int, default=None)
    parse.add_argument("--data", required=True, type=str)
    parse.add_argument("--load", required=True, type=str)
    parse.add_argument("--vgg", required=True, type=str)
    parse.add_argument("--label", required=True, type=str)
    parse.add_argument("--output", required=True, type=str)

    data_fp = parse.parse_args().data
    model_fp = parse.parse_args().load
    vgg_fp = parse.parse_args().vgg
    label_fp = parse.parse_args().label
    output_fp = parse.parse_args().output
    limit = parse.parse_args().l

    pred = prediction(model_fp, vgg_fp, data_fp, label_fp, output_fp, limit)
