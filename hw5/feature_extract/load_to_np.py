import os
import numpy as np

from argparse import ArgumentParser
from reader import getVideoList, readShortVideo




def convert_videos_to_np(mode, labels_fp, videos_fp, save_fp, limit) :
    batch_max = 1000
    l = getVideoList(labels_fp)
    videos_output, labels_output = [], []

    data_num = limit if limit != None else len(l["Video_category"])

    for i in range(data_num):
        print ("Convert videos into numpy: {}/{} \r".format(i + 1, data_num), end="")

        cat = l["Video_category"][i]
        name = l["Video_name"][i]
        label = l["Action_labels"][i]
        data = readShortVideo(videos_fp, cat, name, downsample_factor=12).astype(np.int8)

        videos_output.append(data)
        labels_output.append(int(label))

        if (i+1) % batch_max == 0 :
            videos_output, labels_fp = np.array(videos_output), np.array(labels_output)
            np.save(os.path.join(save_fp, "videos_{}_{}.npy".format(mode, i//batch_max)), videos_output)
            np.save(os.path.join(save_fp, "labels_{}_{}.npy".format(mode, i//batch_max)), labels_output)
            videos_output = []
            labels_output = []

    if (i+1) % batch_max != 0 :
        videos_output, labels_fp = np.array(videos_output), np.array(labels_output)
        np.save(os.path.join(save_fp, "videos_{}_{}.npy".format(mode, (i // batch_max))), videos_output)
        np.save(os.path.join(save_fp, "labels_{}_{}.npy".format(mode, (i // batch_max))), labels_output)

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

    save_fp = parser.parse_args().s
    limit = parser.parse_args().l
    mode = parser.parse_args().m

    convert_videos_to_np(mode, LABEL_PF[mode], VIDEO_PF[mode], save_fp, limit)