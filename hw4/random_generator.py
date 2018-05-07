import cv2
import torch as tor
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.autograd import Variable
from .model import AVE




def random_generator(model, v, output_fp) :
    model.training = False
    model.cuda()
    v = Variable(tor.FloatTensor(v)).cuda()
    img = model.decode(v)
    img = img.cpu().data.numpy() * 255

    plt.imsave(output_fp, img)



if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="model file path")
    parser.add_argument("-o", type=str, required=True, help="output file path")

    model_fp = parser.parse_args().m
    output_fp = parser.parse_args().o

    ave = AVE()
    ave.load_state_dict(tor.load(model_fp))

    rand_v = tor.randn((1, 512))
    random_generator(ave, rand_v, output_fp)