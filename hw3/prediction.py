import os
import cv2
import torch as tor
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from argparse import ArgumentParser

try :
    from model import FCN
except :
    from .model import FCN




def img_recovery(img) :
    """
    :param img: [np.array] shape = (h, w, cha=1) 
    :return: [np.array] shape = (h, w, cha=3)
    """
    pix = [[0, 0, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 255]]
    pix = np.array(pix).astype(np.int16)

    output_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(output_img.shape[0]) :
        for j in range(output_img.shape[1]) :
            output_img[i][j] = pix[int(img[i][j])]

    return output_img



def prediction(model_fp, input_fp, output_fp, limit) :
    model = FCN()
    model.load_state_dict(tor.load(model_fp))
    model.cuda()

    dir_size = len(os.listdir(input_fp))
    limit = limit if limit else float("inf")

    for i in range(dir_size) :
        if i < limit :
            file_name = os.path.join(input_fp, "{:0>4}_sat.jpg".format(i))
            img = plt.imread(file_name)
            #img = plt.imread(file_name)
            img = np.moveaxis(img, 2, 0)
            img = tor.FloatTensor(np.array([img]))
            img_var = Variable(img).type(tor.FloatTensor).cuda()
            pred_img = model(img_var)
            pred_img = tor.max(pred_img, 1)[1]
            pred_img = pred_img.cpu().data.numpy()
            pred_img = np.moveaxis(pred_img, 0, 2)
            output_img = img_recovery(pred_img)
            scipy.misc.imsave(os.path.join(output_fp, "{:0>4}_mask.png".format(i)), output_img)

        else :
            break




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", action="store", type=str, required=True, help="loaded model path")
    parser.add_argument("-l", "--limit", action="store", type=int, default=None, help="limitation of amount of data")
    parser.add_argument("-i", "--input", action="store", type=str, required=True, help="mask pics data file path")
    parser.add_argument("-o", "--output", action="store", type=str, required=True, help="predicted output file path")

    args = parser.parse_args()
    model_fp = args.model
    limit = args.limit
    input_fp = args.input
    output_fp = args.output

    prediction(model_fp, input_fp, output_fp, limit)
