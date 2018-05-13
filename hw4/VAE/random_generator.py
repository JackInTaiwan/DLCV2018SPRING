import time
import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch as tor
from torch.autograd import Variable

try :
    from .model import VAE
    #from .model_2 import AVE
except :
    from model import VAE
    #from model_2 import AVE




def random_generator(model, v, output_fp) :
    model.training = False
    model.cuda()
  
 
    v = Variable(tor.FloatTensor(v)).cuda()
    img = model.decode(v, None) / 2.0 + 0.5
    img = (img.cpu().permute(0, 2, 3, 1).data.numpy()[0] * 255).astype(np.int16)

    img_fn = str(int(time.time())) + ".png"

    plt.imsave(os.path.join(output_fp, img_fn), img)

    print ("|Picture is generated.   |{}".format(img_fn))


if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="model file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file path")

    model_fp = parser.parse_args().model
    output_fp = parser.parse_args().output

    vae = VAE()
    vae.load_state_dict(tor.load(model_fp))

    rand_v = tor.randn((1, 512))
    #rand_v = tor.FloatTensor(1, 512).uniform_(0,1)

    random_generator(vae, rand_v, output_fp)
