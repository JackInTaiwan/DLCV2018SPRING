import os
import cv2
import torch as tor
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable

try :
    from model import VAE
except :
    from .model import VAE




def tsne(dataset_fp, vae_fp) :
    """
    Problem 1-5 plot t-SNE.
    """
    import pandas as pd
    from sklearn.manifold.t_sne import TSNE

    test_num = 100
    batch_size = 50
    testdata_fp = os.path.join(dataset_fp, "test")
    testcsv_fp = os.path.join(dataset_fp, "test.csv")
    attr_selected = "Male"

    vae = VAE()
    vae.training = False
    vae.cuda()
    vae.load_state_dict(tor.load(vae_fp))

    imgs = np.array([])
    latents = np.array([])

    for i in range(40000, 40000 + test_num) :
        img_fp = os.path.join(testdata_fp, "{:0>5}.png".format(i))
        img = plt.imread(img_fp)

        if len(imgs) == 0 :
            imgs = np.array([img])
        else :
            imgs = np.vstack((imgs, np.array([img])))

        if len(imgs) == batch_size :
            imgs_var = (Variable(tor.FloatTensor(imgs)).permute(0, 3, 1, 2).cuda() - 0.5 ) * 2.0
            latent_var, KLD = vae(imgs_var)

            if len(latents) == 0 :
                latents = vae.get_latents().cpu().data.numpy()
            else :
                latents = np.vstack((latents, latent_var.cpu().data.numpy()))

    attr_data = pd.read_csv(testcsv_fp)
    attr_data = np.array(attr_data)[:test_num, list(attr_data.keys()).index(attr_selected)].flatten()
    print ("attr_data shape", attr_data.shape)
    print ("imgs shape", imgs.shape)
    tsne = TSNE(n_components=2)
    latents_tsne = tsne.fit_transform(imgs)
    print ("latents_tsne shape", latents_tsne.shape)




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-q", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    # P 1-5
    parser.add_argument("--vae", type=str)

    q = parser.parse_args().q
    dataset_fp = parser.parse_args().dataset
    vae_fp = parser.parse_args().vae

    if q == "tsne" :
        tsne(dataset_fp, vae_fp)
