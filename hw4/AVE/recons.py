import os
import time
from argparse import ArgumentParser

import numpy as np
import torch as tor
import matplotlib.pyplot as plt
from torch.autograd import Variable


try :
    from model import AVE
    from utils import load_data, console, save_pic, record
except :
    from .model import AVE
    from .utils import load_data, console, save_pic, record




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-i", type=int, required=True, help="Index of picuture")

    model_fp = parser.parse_args().model
    output_fp = parser.parse_args().output
    pic_i = parser.parse_args().i

    model = AVE()
    model.training = False
    model.load_state_dict(tor.load(model_fp))

    pic_fp = os.path.join("../hw4_data/test", "{:0>5}.png".format(pic_i))
    pic = plt.imread(pic_fp)
    print (pic)
    pic = (pic - 0.5) * 2.0
    pic = Variable(tor.FloatTensor(np.array([pic]))).permute(0, 3, 1, 2)
    recon_pic = (model(pic)[0].permute(0, 2, 3, 1).data.numpy()[0] / 2.0 + 0.5) * 255
    recon_pic = recon_pic.astype(np.int16)

    pic_fn = "{}.png".format(int(time.time()))
    plt.imsave(os.path.join(output_fp, pic_fn), recon_pic)

    print ("|Recon Pic is saved {}".format(pic_fn))
