import cv2
import torch as tor
import numpy as np

from argparse import ArgumentParser
from reader import getVideoList, readShortVideo




def prediction(model_fp, data_fp, label_fp) :
    model = tor.load(model_fp)
    model.cuda()

    ### Load data
    l = getVideoList(label_fp)
    videos_output, labels_output = [], []

    for i in range(len(l["Video_category"])):
        print("Convert videos into numpy: {}/{} \r".format(i + 1, len(l["Video_category"])), end="")

        cat = l["Video_category"][i]
        name = l["Video_name"][i]
        label = l["Action_labels"][i]
        data = readShortVideo(data_fp, cat, name, downsample_factor=12).astype(np.int8)

        videos_output.append(data)
        labels_output.append(int(label))


    videos, labels = np.array(videos_output), np.array(labels_output).astype(np.uint8)


    ### Prediction
    correct, total = 0, len(labels)

    for i, (x, label) in enumerate(zip(videos, labels), 1) :
        print ("Process: {}/{}".format(i, len(videos)))
        x = tor.Tensor(x).permute(0, 3, 1, 2).cuda()
        out = model(x)
        out = out.mean(dim=0).unsqueeze(0)
        pred = model.pred(out)
        y = tor.max(pred, 1)[1]
        if int(y[0].data) == label :
            correct += 1

    acc = correct / total
    print (acc)




if __name__ == "__main__" :
    parse = ArgumentParser()
    parse.add_argument("--data", required=True)
    parse.add_argument("--load", required=True, type=str)
    parse.add_argument("--label", required=True, type=str)

    data_fp = parse.parse_args().data
    model_fp = parse.parse_args().load
    label_fp = parse.parse_args().label

    pred = prediction(model_fp, data_fp, label_fp)