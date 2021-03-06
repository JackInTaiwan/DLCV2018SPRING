import os
import math
import torchvision.models
import numpy as np
import torch as tor

from argparse import ArgumentParser
from reader import getVideoList, readShortVideo
from torchvision.transforms import Normalize




""" Parameter """
MAX_VIDEO_LEN = 10




""" vgg16 model"""
class Vgg16(tor.nn.Module) :
    def __init__(self) :
        super(Vgg16, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.eval()

        self.vgg16 = vgg16.features


    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)

        return x




""" Functions """
def convert_videos_to_np(mode, labels_fp, videos_fp, save_fp, limit, model) :
    batch_max = 1000
    l = getVideoList(labels_fp)
    videos_output, labels_output = [], []

    data_num = limit if limit != None else len(l["Video_category"])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = Normalize(mean, std)


    for batch in range(data_num):
        print ("Convert videos into numpy: {}/{} \r".format(batch + 1, data_num), end="")

        cat = l["Video_category"][batch]
        name = l["Video_name"][batch]
        label = l["Action_labels"][batch]
        data = readShortVideo(videos_fp, cat, name, downsample_factor=12)


        if len(data) > MAX_VIDEO_LEN :
            seq = [math.floor(data.shape[0] * _i / MAX_VIDEO_LEN) for _i in range(MAX_VIDEO_LEN)]
            data = data[seq]

        data = tor.Tensor(data).permute(0, 3, 1, 2) / 255.


        for i in range(len(data)) :
            data[i] = norm(data[i])
        data = data.cuda()
        out = model(data)
        features = out.cpu().data.numpy()
        videos_output.append(features)
        labels_output.append(int(label))


        if (batch+1) % batch_max == 0 :
            videos_output, labels_fp = np.array(videos_output), np.array(labels_output)
            np.save(os.path.join(save_fp, "videos_{}_{}.npy".format(mode, batch//batch_max)), videos_output)
            np.save(os.path.join(save_fp, "labels_{}_{}.npy".format(mode, batch//batch_max)), labels_output)
            videos_output = []
            labels_output = []


    if (batch+1) % batch_max != 0 :
        videos_output, labels_fp = np.array(videos_output), np.array(labels_output)
        np.save(os.path.join(save_fp, "videos_{}_{}.npy".format(mode, (batch // batch_max))), videos_output)
        np.save(os.path.join(save_fp, "labels_{}_{}.npy".format(mode, (batch // batch_max))), labels_output)


    print ("\nDone !")





if __name__ == "__main__" :
    LABEL_PF = {
        "train": "../HW5_data/TrimmedVideos/label/gt_train.csv",
        "valid": "../HW5_data/TrimmedVideos/label/gt_valid.csv",
    }
    VIDEO_PF = {
        "train": "../HW5_data/TrimmedVideos/video/train",
        "valid": "../HW5_data/TrimmedVideos/video/valid",
    }

    parser = ArgumentParser()
    parser.add_argument("-m", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("-s", type=str, default="", help="Saving file path")
    parser.add_argument("-l", type=int, default=None, help="Limit of amount of required data")
    parser.add_argument("--load", type=str, default=None, help="loaded model file path")

    save_fp = parser.parse_args().s
    limit = parser.parse_args().l
    mode = parser.parse_args().m
    model_fp = parser.parse_args().load

    if model_fp :
        model = tor.load(model_fp)
    else :
        model = Vgg16()

    model.cuda()

    convert_videos_to_np(mode, LABEL_PF[mode], VIDEO_PF[mode], save_fp, limit, model)
