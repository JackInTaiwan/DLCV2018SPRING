import cv2
import os
import torch as tor
import numpy as np

from argparse import ArgumentParser
from reader import getVideoList, readShortVideo
from utils import normalize, select_data
from train import VIDEOS_MAX_BATCH




def prediction(model_fp, data_fp, label_fp, output_fp, limit) :
    model = tor.load(model_fp)
    model.cuda()

    ### Load data
    l = getVideoList(label_fp)
    videos_output, labels_output = [], []

    total = len(l["Video_category"]) if not limit else limit

    for i in range(total):
        print("Convert videos into numpy: {}/{} \r".format(i + 1, len(l["Video_category"])), end="")

        cat = l["Video_category"][i]
        name = l["Video_name"][i]
        label = l["Action_labels"][i]
        data = readShortVideo(data_fp, cat, name, downsample_factor=12).astype(np.int8)
        videos_output.append(data.astype(np.int16))
        labels_output.append(int(label))


    videos, labels = np.array(videos_output), np.array(labels_output).astype(np.uint8)


    ### Prediction
    correct, total = 0, len(labels)
    preds = []

    videos = normalize(videos / 255.)
    videos = select_data(videos, VIDEOS_MAX_BATCH)

    for i, (x, label) in enumerate(zip(videos, labels), 1) :
        print ("Process: {}/{}".format(i, len(videos)))
        x = tor.Tensor(x).permute(0, 3, 1, 2).cuda()
        out = model(x)
        out = out.mean(dim=0).unsqueeze(0)
        pred = model.pred(out)
        y = tor.max(pred, 1)[1]
        pred = int(y[0].data)
        if pred == label :
            correct += 1

        preds.append(pred)

    acc = correct / total
    print (acc)

    with open(os.path.join(output_fp, "p1_valid.txt"), "w") as f :
        for i, item in enumerate(preds) :
            if i != len(preds)-1 :
                f.write(str(item) + "\n")
            else :
                f.write(str(item))





if __name__ == "__main__" :
    parse = ArgumentParser()
    parse.add_argument("-l", type=int, default=None)
    parse.add_argument("--data", required=True, type=str)
    parse.add_argument("--load", required=True, type=str)
    parse.add_argument("--label", required=True, type=str)
    parse.add_argument("--output", required=True, type=str)

    data_fp = parse.parse_args().data
    model_fp = parse.parse_args().load
    label_fp = parse.parse_args().label
    output_fp = parse.parse_args().output
    limit = parse.parse_args().l

    pred = prediction(model_fp, data_fp, label_fp, output_fp, limit)
