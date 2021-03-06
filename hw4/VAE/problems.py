import os
import cv2
import torch as tor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable




def tsne(dataset_fp, model_fp, out_fp) :
    """
    Problem 1-5 plot t-SNE.
    """
    import pandas as pd
    from sklearn.manifold.t_sne import TSNE

    try :
        from model_fin import VAE
    except :
        from .model_fin import VAE

    test_num = 500
    batch_size = 50
    testdata_fp = os.path.join(dataset_fp, "test")
    testcsv_fp = os.path.join(dataset_fp, "test.csv")
    attr_selected = "Blond_Hair"

    vae = VAE()
    vae.training = False
    vae.cuda()
    vae.load_state_dict(tor.load(model_fp))

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

    tsne = TSNE(
        n_components=2,
        random_state=0,
    )
    latents_tsne = tsne.fit_transform(latents)

    plt.scatter(latents_tsne[attr_data == 0, 0], latents_tsne[attr_data == 0, 1], c="r")
    plt.scatter(latents_tsne[attr_data == 1, 0], latents_tsne[attr_data == 1, 1], c="b")
    plt.legend(["Not {}".format(attr_selected), attr_selected])

    plt.savefig(os.path.join(out_fp, "fig1_5.jpg"))




def lcurve(record_fp, out_fp) :
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
    import torch.nn.functional as F

    try :
        from model_fin import VAE
    except :
        from .model_fin import VAE

    tor.manual_seed(0)
    generate_num = 32
    latent_size = 512

    model = VAE()
    model.training = False
    model.cuda()
    model.load_state_dict(tor.load(model_fp))

    x = Variable(F.tanh(tor.randn(generate_num, latent_size))).cuda()

    imgs = model.decode(x, None)
    imgs = ((imgs.permute(0, 2, 3, 1).cpu().data.numpy() / 2.0 ) + 0.5) * 255
    imgs = imgs.astype(np.int16)

    for i, img in enumerate(imgs, 1) :
        plt.subplot(4, 8, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(output_fp, "fig1_4.jpg"))




def test_plot(dataset_fp, model_fp, out_fp) :
    try :
        from model_fin import VAE
    except :
        from .model_fin import VAE

    model = VAE()
    model.training = False
    model.cuda()
    model.load_state_dict(tor.load(model_fp))

    imgs = np.array([])

    for i in range(10) :
        img = plt.imread(os.path.join(dataset_fp, "test", "{:0>5}.png".format(40000 + i)))
        if len(imgs) == 0 :
            imgs = np.array([img])
        else :
            imgs = np.vstack((imgs, np.array([img])))

    imgs_var = (Variable(tor.FloatTensor(imgs)).permute(0, 3, 1, 2) - 0.5 ) * 2.0
    imgs_var = imgs_var.cuda()

    imgs_recon, KLD = model(imgs_var)
    imgs_recon = imgs_recon.permute(0, 2, 3, 1).cpu().data.numpy()
    imgs_recon = (imgs_recon / 2.0) + 0.5

    for i, img in enumerate(imgs, 1) :
        plt.subplot(2, 10, i)
        plt.xticks([])
        plt.yticks([])
        #plt.title("{:0>5}.png".format(40000 + i - 1))
        plt.imshow(img)

    for i, img in enumerate(imgs_recon, 1) :
        plt.subplot(2, 10, i + 10)
        plt.xticks([])
        plt.yticks([])
        #plt.title("{:0>5}.png".format(40000 + i - 1))
        plt.imshow(img)


    plt.tight_layout(pad=0.0, h_pad=-10.0)
    plt.savefig(os.path.join(out_fp, "fig1_3.jpg"))




def test_loss(dataset_fp, model_fp) :
    try :
        from model_fin import VAE
    except :
        from .model_fin import VAE

    model = VAE()
    model.training = False
    model.cuda()
    model.load_state_dict(tor.load(model_fp))

    loss_func = tor.nn.MSELoss().cuda()

    imgs = np.array([])
    batchsize = 8

    total_loss = np.array([])

    for i, fn in enumerate(os.listdir(os.path.join(dataset_fp, "test")), 1) :
        img = plt.imread(os.path.join(dataset_fp, "test", fn))

        if len(imgs) == 0 :
            imgs = np.array([img])
        else :
            imgs = np.vstack((imgs, np.array([img])))

        if i % batchsize == 0 :
            imgs_var = (Variable(tor.FloatTensor(imgs)).permute(0, 3, 1, 2).cuda() - 0.5) * 2.0
            output, KLD = model(imgs_var)
            loss = loss_func(output, imgs_var)
            total_loss =  np.concatenate((total_loss, [float(loss.cpu().data)]))
            #print ("loss:", float(loss.cpu().data))

        imgs = np.array([])

    print ("Average total loss:", total_loss.mean())




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-q", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--model", type=str)
    # P 1-1
    parser.add_argument("--record", type=str)

    q = parser.parse_args().q
    dataset_fp = parser.parse_args().dataset
    model_fp = parser.parse_args().model
    out_fp = parser.parse_args().output
    record_fp = parser.parse_args().record

    if q == "tsne" :
        tsne(dataset_fp, model_fp, out_fp)

    elif q == "lcurve" :
        lcurve(record_fp, out_fp)

    elif q == "test" :
        test_plot(dataset_fp, model_fp, out_fp)

    elif q == "rg" :
        rand_generator(out_fp, model_fp)

    elif q == "test_loss" :
        test_loss(dataset_fp, model_fp)
