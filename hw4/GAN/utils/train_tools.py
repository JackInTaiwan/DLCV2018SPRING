import numpy as np



def load_data(x_fp) :
    x = np.load(x_fp)
    x_size = x.shape

    return x, x_size




def save_pic(save_fp, model, pic_n) :
    import cv2
    import os
    import time
    import matplotlib.pyplot as plt
    import torch as tor
    from torch.autograd import Variable

    tor.manual_seed(0)

    for i in range(pic_n) :
        img = tor.randn(1, 512)
        img_var = Variable(img).cuda()
        out = model(img_var)

        out = out.permute(0, 2, 3, 1).cpu()
        out_img = (out.data.numpy()[0] / 2.0 + 0.5) * 255

        f = "{}_{:0>5}.png".format(int(time.time()), i)
        plt.imsave(os.path.join(save_fp, f), out_img.astype(np.int16))

        print ("|Output {} is saved.".format(f))




if __name__ == "__main__" :
    import torch as tor
    from model_GAN import GN
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-n", type=int, default=1)

    model_fp = parser.parse_args().model
    save_fp = parser.parse_args().output
    n = parser.parse_args().n

    model = GN()
    model.cuda()
    model.load_state_dict(tor.load(model_fp))
    save_pic(save_fp, model, n)
