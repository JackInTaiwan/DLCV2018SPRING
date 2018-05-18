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


    def get_avg(arr, avg_num) :
        for i in range(len(arr)) :
            if i < (avg_num // 2) or i >= len(arr) - (avg_num // 2) :
                arr[i] = arr[i]
            else :
                arr[i] = arr[i-(avg_num//2): i+(avg_num//2)].mean()
        return arr


    with open(record_fp, "r") as f :
        data = json.load(f)

    data = json.loads(data)


    ### Loss
    avg_num = 9
    loss_fake = np.array(data["loss_fake"])
    loss_real = np.array(data["loss_real"])
    loss_fake_avg = get_avg(np.copy(loss_fake), avg_num)
    loss_real_avg = get_avg(np.copy(loss_real), avg_num)
    record_step = 10
    steps_fake = np.array(range(len(loss_fake))) * record_step
    steps_real = np.array(range(len(loss_real))) * record_step


    plt.subplot(2, 1, 1)
    plt.plot(steps_real, loss_real, linewidth=0.5, c="#ffaa0033")
    plt.plot(steps_fake, loss_fake, linewidth=0.5, c="#00aaff33")
    plt.plot(steps_real[avg_num:-avg_num], loss_real_avg[avg_num:-avg_num], linewidth=0.5, c="#ffaa00")
    plt.plot(steps_fake[avg_num:-avg_num], loss_fake_avg[avg_num:-avg_num], linewidth=0.5, c="#00aaff")
    plt.legend(["Real", "Avg. Real", "Fake", "Avg. Fake"])
    plt.xlabel("Steps")
    plt.ylabel("Loss")


    ## Accuracy
    acc_true = np.array(data["acc_true"])
    acc_false = np.array(data["acc_false"])
    avg_num = 9
    acc_true_avg = get_avg(np.array(acc_true), avg_num)
    acc_false_avg = get_avg(np.array(acc_false), avg_num)
    record_step = 10
    steps = np.array(range(len(acc_true_avg))) * record_step


    plt.subplot(2, 1, 2)
    plt.plot(steps, acc_true, linewidth=1.0, c="#0088ff33")
    plt.plot(steps, acc_false, linewidth=1.0, c="#ff000033")
    plt.plot(steps[avg_num: -avg_num], acc_true_avg[avg_num: -avg_num], linewidth=1.0, c="#0088ff")
    plt.plot(steps[avg_num: -avg_num], acc_false_avg[avg_num: -avg_num], linewidth=1.0, c="#ff0000")
    plt.legend(["Real", "Fake", "Avg. Real", "Avg. Fake"])
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(out_fp, "fig3_2.jpg"))




def rand_generator(output_fp, model_fp) :
    import torch as tor
    from torch.autograd import Variable

    try :
        from model_3 import GN
    except :
        from .model_3 import GN
    for s in range(10) :
        tor.manual_seed(s)
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
        imgs = ((imgs.permute(0, 2, 3, 1).cpu().data.numpy() / 2.0) + 0.5) * 255
        imgs = imgs.astype(np.int16)

        for i, img in enumerate(imgs, 1) :
            plt.subplot(2, 10, i+10)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)

        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(output_fp, "fig3_3_{}.jpg".format(s)))




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

    if q == "lcurve" :
        lcurve(record_fp, out_fp)

    elif q == "rg" :
        rand_generator(out_fp, model_fp)
