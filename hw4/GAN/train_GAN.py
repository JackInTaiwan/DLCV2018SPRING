import cv2
import os
from argparse import ArgumentParser

import numpy as np
import torch as tor
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from model_GAN import GN, DN
    from utils import load_data, console, save_pic, record
except:
    from .model_GAN import GN, DN
    from .utils import load_data, console, save_pic, record





""" Parameters """
AVAILABLE_SIZE = None
EPOCH = 30
BATCHSIZE = 64
EVAL_SIZE = 64
PIVOT_STEPS = 30

LR, LR_STEPSIZE, LR_GAMMA = 0.0001, 500, 0.95
MOMENTUM = 0.5

RECORD_JSON_PERIOD = 10  # steps
RECORD_MODEL_PERIOD = 2  # epochs
RECORD_PIC_PERIOD = 60  # steps

TRAIN_DATA_FP = ["../data/train_data.npy", "../data/train_data_1.npy", "../data/train_data_2.npy"]

RECORD_FP = "./record/model_gan.json"

MODEL_ROOT = "./models"




""" Data Setting """
def data_loader(limit):
    x_train_1, x_size_1 = load_data(TRAIN_DATA_FP[0])
    x_train_2, x_size_2 = load_data(TRAIN_DATA_FP[1])
    x_train_3, x_size_3 = load_data(TRAIN_DATA_FP[2])
    x_train = np.vstack((x_train_1, x_train_2, x_train_3))
    """
    x_train, x_size = load_data(TRAIN_DATA_FP[0])
    """
    print(x_train.shape)

    if limit:
        x_train = x_train[:limit]

    global AVAILABLE_SIZE
    AVAILABLE_SIZE = x_train.shape

    x_train = tor.FloatTensor(x_train).permute(0, 3, 1, 2)
    x_eval_train = x_train[:EVAL_SIZE]

    data_set = TensorDataset(
        data_tensor=x_train,
        target_tensor=x_train
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
    )

    return data_loader, x_eval_train


def save_record(model_index, epoch, optim, loss, acc_true, acc_false):
    record_data = dict()

    if epoch == 0:
        record_data["model_name"] = "gan_model_{}.pkl".format(model_index)
        record_data["data_size"] = AVAILABLE_SIZE
        record_data["batch_size"] = BATCHSIZE
        record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
        record_data["lr_init"] = float(optim.param_groups[0]["lr"])
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["record_epoch"] = RECORD_MODEL_PERIOD
        record_data["loss"] = round(float(loss.data), 6)


    else:
        record_data["model_name"] = "gan_model_{}.pkl".format(model_index)
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["loss"] = round(float(loss.data), 6)

    record(RECORD_FP, record_data)


""" Model Training """


def train(data_loader, model_index, x_eval_train, gn_fp, dn_fp, ave_fp):
    ### Model Initiation
    gn = GN().cuda()
    dn = DN().cuda()

    ave_state_dict = tor.load(ave_fp)
    gn.load_ave_state(ave_state_dict)
    dn.load_ave_state(ave_state_dict)

    if gn_fp :
        gn_state_dict = tor.load(gn_fp)
        gn.load_state_dict(gn_state_dict)
    if dn_fp :
        dn_state_dict = tor.load(dn_fp)
        dn.load_state_dict(dn_state_dict)


    loss_func = tor.nn.BCELoss()

    #optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
    optim_gn = tor.optim.Adam(gn.parameters(), lr=LR)
    optim_dn = tor.optim.Adam(dn.parameters(), lr=LR)

    lr_step_gn = StepLR(optim_gn, step_size=LR_STEPSIZE, gamma=LR_GAMMA)
    lr_step_dn = StepLR(optim_dn, step_size=LR_STEPSIZE, gamma=LR_GAMMA)


    ### Training
    for epoch in range(EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1))

        for step, (x_batch, y_batch) in enumerate(data_loader):
            print("Process: {}/{}".format(step, int(AVAILABLE_SIZE[0] / BATCHSIZE)), end="\r")

            ### train true/false pic
            if (step // PIVOT_STEPS) % 2 == 0 :
                out = Variable(x_batch).cuda() if step % 2 == 0 else gn(Variable(tor.randn(BATCHSIZE, 512)).cuda())
                ans = Variable(tor.ones(BATCHSIZE)).cuda() if step % 2 == 0 else Variable(tor.zeros(BATCHSIZE)).cuda()
                dis = dn(out)

                optim = optim_dn

            else :
                out = gn(Variable(tor.randn(BATCHSIZE, 512)).cuda())
                ans = Variable(tor.ones(BATCHSIZE)).cuda()
                dis = dn(out)

                optim = optim_gn

            loss = loss_func(dis, ans)
            loss.backward()
            optim.step()
            optim.zero_grad()
            lr_step_dn.step()
            lr_step_gn.step()


            if step % RECORD_JSON_PERIOD == 0 :
                x_true = Variable(x_eval_train).cuda()
                out = dn(x_true)
                acc_true = round(int((out >= 0.5).sum().data) / EVAL_SIZE, 5)

                x_false = Variable(tor.randn((EVAL_SIZE, 3, 64, 64))).cuda()
                out = dn(x_false)
                acc_false = round(int((out < 0.5).sum().data) / EVAL_SIZE, 5)

                print ("|Acc True: {}   |Acc False: {}".format(acc_true, acc_false))

                save_record(model_index, epoch, optim, loss, acc_true, acc_false)

            if step % RECORD_PIC_PERIOD == 0 :
                loss = float(loss.data)
                print("|Loss: {:<8}".format(loss))
                save_pic("output", gn, 3)

            if step % (2 * PIVOT_STEPS) == 0 :
                pass


        ### Save model
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(gn.state_dict(), os.path.join(MODEL_ROOT, "gan_gn_{}_{}.pkl".format(model_index, epoch)))
            tor.save(dn.state_dict(), os.path.join(MODEL_ROOT, "gan_dn_{}_{}.pkl".format(model_index, epoch)))


""" Main """
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("--gn", action="store", type=str, default=None)
    parser.add_argument("--dn", action="store", type=str, default=None)
    parser.add_argument("--ave", action="store", type=str, required=True, help="pretrained AVE model file path")

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i
    gn_fp = parser.parse_args().gn
    dn_fp = parser.parse_args().dn
    ave_fp = parser.parse_args().ave
    LR = parser.parse_args().lr if parser.parse_args().lr else LR

    ### Load Data
    console("Load Data")
    data_loader, x_eval_train = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index, x_eval_train, gn_fp, dn_fp, ave_fp)


