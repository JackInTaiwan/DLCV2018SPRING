import os
import math
import torchvision.models
import numpy as np
import torch as tor
import matplotlib.pyplot as plt

from argparse import ArgumentParser
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
def read_pics(fp) :
    output = np.empty((len(os.listdir(fp)), 240, 320, 3))

    for i, pic_fn in enumerate(os.listdir(fp)) :
        pic_fp = os.path.join(fp, pic_fn)
        pic = plt.imread(pic_fp)
        output[i] = pic

    return output



def read_labels(fp) :
    labels = []

    with open(fp) as f :
        for line in f :
            labels.append(int(line))
    labels = np.array(labels).astype(np.int8)

    return labels



def convert_videos_to_np(mode, labels_fp, videos_fp, save_fp, limit, model) :
    videos_output, labels_output = [], []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = Normalize(mean, std)


    for video_fn in os.listdir(videos_fp) :
        #print ("Convert videos into numpy: {}/{} \r".format(batch + 1, data_num), end="")

        data = read_pics(os.path.join(videos_fp, video_fn))

        data = tor.Tensor(data).permute(0, 3, 1, 2) / 255.

        for i in range(len(data)) :
            data[i] = norm(data[i])

        video_stack = np.empty((len(data), 1024))
        data = data.cuda()

        for i, datum in enumerate(data) :
            print (datum.size())
            datum.cuda()
            out = model(datum)
            features = out.cpu().data.numpy()
            video_stack[i] = features

        videos_output.append(features)


        label_stack = read_labels(os.path.join(labels_fp, "{}.txt".format(video_fn)))
        labels_output.append(label_stack)



        videos_output, labels_fp = np.array(videos_output), np.array(labels_output)
        np.save(os.path.join(save_fp, "videos_{}.npy".format(mode)), videos_output)
        np.save(os.path.join(save_fp, "labels_{}.npy".format(mode)), labels_output)

    print ("\nDone !")





if __name__ == "__main__" :
    LABEL_PF = {
        "train": "../HW5_data/FullLengthVideos/labels/train",
        "valid": "../HW5_data/FullLengthVideos/labels/valid",
    }
    VIDEO_PF = {
        "train": "../HW5_data/FullLengthVideos/videos/train",
        "valid": "../HW5_data/FullLengthVideos/videos/valid",
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
    model = None
    """
    if model_fp :
        model = tor.load(model_fp)
    else :
        model = Vgg16()
    """
    #model.cuda()


    convert_videos_to_np(mode, LABEL_PF[mode], VIDEO_PF[mode], save_fp, limit, model)
