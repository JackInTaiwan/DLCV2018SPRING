import os
import cv2
import torch as tor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable




def lcurve(record_fp, out_fp) :
    import json

    with open(record_fp, "r") as f :
        data = json.load(f)

    data = json.loads(data)


    ### Loss
    loss_fake = np.array(data["loss_fake"])
    loss_real = np.array(data["loss_real"])
    loss_fake_avg = np.copy(loss_fake)
    loss_real_avg = np.copy(loss_real)
    record_step = 10
    steps_fake = np.array(range(len(loss_fake))) * record_step
    steps_real = np.array(range(len(loss_real))) * record_step
    avg_num = 9
    """
    for i in range(len(loss_avg)) :
        if i < (avg_num // 2) or i >= len(loss_avg) - (avg_num // 2) :
            loss_avg[i] = loss[i]
        else :
            loss_avg[i] = loss[i-(avg_num//2): i+(avg_num//2)].mean()

    plt.subplot(2, 1, 1)
    plt.plot(steps, loss, linewidth=0.5)
    plt.plot(steps, loss_avg, linewidth=0.5, c="r")
    plt.legend(["Original", "Averaged"])
    plt.xlabel("Steps")
    plt.ylabel("Loss")


    ## Accuracy
    acc_true = np.array(data["acc_true"])
    acc_false = np.array(data["acc_false"])
    acc_true_avg = np.copy(acc_true)
    acc_false_avg = np.array(acc_false)
    record_step = 10
    steps = np.array(range(len(loss))) * record_step
    avg_num = 9

    for i in range(len(acc_true_avg)) :
        if i < (avg_num // 2) or i >= len(acc_true_avg) - (avg_num // 2) :
            acc_true_avg[i] = acc_true[i]
        else :
            acc_true_avg[i] = acc_true[i-(avg_num//2): i+(avg_num//2)].mean()

    for i in range(len(acc_false_avg)) :
        if i < (avg_num // 2) or i >= len(acc_false) - (avg_num // 2) :
            acc_false_avg[i] = acc_false[i]
        else :
            acc_false_avg[i] = acc_false[i-(avg_num//2): i+(avg_num//2)].mean()

    plt.subplot(2, 1, 2)
    plt.plot(steps, acc_true, linewidth=1.0, c="#0088ff33")
    plt.plot(steps, acc_false, linewidth=1.0, c="#ff000033")
    plt.plot(steps[avg_num: -avg_num], acc_true_avg[avg_num: -avg_num], linewidth=1.0, c="#0088ff")
    plt.plot(steps[avg_num: -avg_num], acc_false_avg[avg_num: -avg_num], linewidth=1.0, c="#ff0000")
    plt.legend(["Real", "Fake", "Avg. Real", "Avg. Fake"])
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(out_fp, "fig2_2.jpg"))
    """




def rand_generator(output_fp, model_fp) :
    import torch as tor
    from torch.autograd import Variable

    try :
        from model_3 import GN
    except :
        from .model_3 import GN

    tor.manual_seed(0)
    generate_num = 10
    latent_size = 512

    model = GN()
    model.cuda()
    model.load_state_dict(tor.load(model_fp))

    xs = Variable(tor.randn(generate_num, latent_size)).cuda()
    xs[:, 0] = 0

    imgs = model(xs)

    imgs = ((imgs.permute(0, 2, 3, 1).cpu().data.numpy() / 2.0) + 0.5) * 255
    imgs = imgs.astype(np.int16)

    for i, img in enumerate(imgs, 1) :
        plt.subplot(2, 10, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    xs[:, 0] = 1

    imgs = model(xs)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().data.numpy() / 2.0) + 0.5
    imgs = imgs.astype(np.int16)

    for i, img in enumerate(imgs, 1) :
        plt.subplot(2, 10, i+10)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(output_fp, "fig3_3.jpg"))




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
