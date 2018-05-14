import os
import cv2
import torch as tor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable

try :
    from model import VAE
except :
    from .model import VAE





def tsne(dataset_fp, vae_fp, out_fp) :
    """
    Problem 1-5 plot t-SNE.
    """
    import pandas as pd
    from sklearn.manifold.t_sne import TSNE

    test_num = 500
    batch_size = 50
    testdata_fp = os.path.join(dataset_fp, "test")
    testcsv_fp = os.path.join(dataset_fp, "test.csv")
    attr_selected = "Smiling"

    vae = VAE()
    vae.training = False
    vae.cuda()
    vae.load_state_dict(tor.load(vae_fp))

    imgs = np.array([])
    latents = np.array([])
    imgs_tmp = np.array([])

    for i in range(40000, 40000 + test_num) :
        img_fp = os.path.join(testdata_fp, "{:0>5}.png".format(i))
        img = plt.imread(img_fp)

        if len(imgs_tmp) == 0 :
            imgs_tmp = np.array([img])
        else :
            imgs_tmp = np.vstack((imgs_tmp, np.array([img])))

        if len(imgs_tmp) == batch_size :
            imgs_var = (Variable(tor.FloatTensor(imgs_tmp)).permute(0, 3, 1, 2).cuda() - 0.5 ) * 2.0
            latent_var, KLD = vae(imgs_var)

            if len(latents) == 0 :
                latents = vae.get_latents().cpu().data.numpy()
            else :
                latents = np.vstack((latents, vae.get_latents().cpu().data.numpy()))

            imgs_tmp = np.array([])

    attr_data = pd.read_csv(testcsv_fp)
    attr_data = np.array(attr_data)[:test_num, list(attr_data.keys()).index(attr_selected)].flatten()

    tsne = TSNE(n_components=2)
    latents_tsne = tsne.fit_transform(latents)

    plt.scatter(latents[attr_data == 0, 0], latents[attr_data == 0, 1], c="r")
    plt.scatter(latents[attr_data == 1, 0], latents[attr_data == 1, 1], c="b")
    plt.legend(["Not {}".format(attr_selected), attr_selected])

    plt.savefig(os.path.join(out_fp, "fig1_5.jpg"))




def lcurve(record_fp, output_fp) :
    import json

    with open(record_fp, "r") as f :
        data = json.load(f)

    data = json.loads(data)
    recon_loss = data["recon_loss"]
    KLD_loss = data["KLD_loss"]
    record_step = 10
    steps = np.array(range(len(recon_loss))) * record_step

    plt.subplot(2, 1, 1)
    plt.plot(steps, recon_loss, linewidth=0.5,)
    plt.xlabel("Steps")
    plt.ylabel("Recon. Loss")

    plt.subplot(2, 1, 2)
    plt.plot(steps, KLD_loss, linewidth=0.5)
    plt.xlabel("Steps")
    plt.ylabel("KLD Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(out_fp, "fig1_2.jpg"))




def rand_generator(output_fp, model_fp) :
    import torch as tor
    from torch.autograd import Variable

    try :
        from model_2 import VAE
    except :
        from .model_2 import VAE

    tor.manual_seed(0)
    generate_num = 32
    latent_size = 512

    model = VAE()
    model.training = False
    model.cuda()
    model.load_state_dict(tor.load(model_fp))

    x = Variable(tor.randn(generate_nugam, latent_size)).cuda()

    imgs = model.decode(x, None)
    imgs = ((imgs.permute(0, 2, 3, 1).cuda().data.nump() / 2.0 ) + 0.5) * 255
    imgs = imgs.astype(np.int16)

    for i, img in enumerate(imgs, 1) :
        plt.subplot(4, 8, i)
        plt.imshow(img)

    plt.savefig(os.path.join(output_fp, "fig1_4.jpg"))




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-q", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    # P 1-5
    parser.add_argument("--vae", type=str)
    # P 1-1
    parser.add_argument("--record", type=str)
    # P 1-4
    parser.add_argument("--model", type=str)

    q = parser.parse_args().q
    dataset_fp = parser.parse_args().dataset
    vae_fp = parser.parse_args().vae
    out_fp = parser.parse_args().output
    record_fp = parser.parse_args().record
    model_fp = parser.parse_args().model

    if q == "tsne" :
        tsne(dataset_fp, vae_fp, out_fp)

    elif q == "lcurve" :
        lcurve(record_fp, out_fp)

    elif q == "rg" :
        rand_generator(out_fp, model_fp)