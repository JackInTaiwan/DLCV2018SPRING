import cv2
import os
import time
import numpy as np
import pandas as pd
import torch as tor
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from model_2 import GN, DN
    from utils import load_data, console, save_pic_2, record
except:
    from .model_2 import GN, DN
    from .utils import load_data, console, save_pic_2, record

""" Parameters """
AVAILABLE_SIZE = None
EPOCH = 50
BATCHSIZE = 32
EVAL_SIZE = 64
PIVOT_STEPS = 50

LR, LR_STEPSIZE, LR_GAMMA = 0.0001, 2000, 0.95
MOMENTUM = 0.5

LATENT_SPACE = 512

RECORD_JSON_PERIOD = 50  # steps
RECORD_MODEL_PERIOD = 1000  # steps
RECORD_PIC_PERIOD = 360  # steps

TRAIN_DATA_FP = ["../data/train_data.npy", "../data/train_data_1.npy", "../data/train_data_2.npy"]
ATTR_DATA_FP = "../hw4_data/train.csv"
SELECTED_ATTR = "Male"

RECORD_FP = "./record"

MODEL_ROOT = "./models"

""" Data Setting """


def data_loader(limit):
    ### Load img data
    x_train_1, x_size_1 = load_data(TRAIN_DATA_FP[0])
    x_train_2, x_size_2 = load_data(TRAIN_DATA_FP[1])
    x_train_3, x_size_3 = load_data(TRAIN_DATA_FP[2])
    x_train = np.vstack((x_train_1, x_train_2, x_train_3))

    ### Load attribute data
    data_attr = pd.read_csv(ATTR_DATA_FP)
    attr_train = np.array(data_attr)[:, list(data_attr.keys()).index(SELECTED_ATTR)]
    attr_train = attr_train.reshape(-1, 1).astype(np.int64)

    if limit:
        x_train, attr_train = x_train[:limit], attr_train[:limit]

    global AVAILABLE_SIZE
    AVAILABLE_SIZE = x_train.shape

    x_train = tor.FloatTensor(x_train).permute(0, 3, 1, 2)
    attr_train = tor.FloatTensor(attr_train)
    x_eval_train = x_train[:EVAL_SIZE]

    data_set = TensorDataset(
        data_tensor=x_train,
        target_tensor=attr_train
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
    )

    return data_loader, x_eval_train


def save_record(model_index, epoch, optim, loss_real, loss_fake, acc_true, acc_false):
    record_data = dict()

    model_name = "acgan_{}.pkl".format(model_index)

    if epoch == 0:
        record_data["model_name"] = model_name
        record_data["data_size"] = AVAILABLE_SIZE
        record_data["batch_size"] = BATCHSIZE
        record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
        record_data["lr_init"] = float(optim.param_groups[0]["lr"])
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["record_epoch"] = RECORD_MODEL_PERIOD
        record_data["loss_real"] = round(float(loss_real.data), 6)
        record_data["loss_fake"] = round(float(loss_fake.data), 6)
        record_data["acc_true"] = acc_true
        record_data["acc_false"] = acc_false


    else:
        record_data["model_name"] = model_name
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["loss_real"] = round(float(loss_real.data), 6)
        record_data["loss_fake"] = round(float(loss_fake.data), 6)
        record_data["acc_true"] = acc_true
        record_data["acc_false"] = acc_false

    json_fp = os.path.join(RECORD_FP, "acgan_{}.json".format(model_index))

    if not os.path.exists(json_fp):
        with open(json_fp, "w") as f:
            f.write('"[]"')

    record(json_fp, record_data)


""" Model Training """


def train(data_loader, model_index, x_eval_train, gn_fp, dn_fp, gan_gn_fp, gan_dn_fp):
    ### Model Initiation
    gn = GN().cuda()
    dn = DN().cuda()

    if gn_fp:
        gn_state_dict = tor.load(gn_fp)
        gn.load_state_dict(gn_state_dict)
    if dn_fp:
        dn_state_dict = tor.load(dn_fp)
        dn.load_state_dict(dn_state_dict)
    if gan_dn_fp:
        dn.load_dn_state(tor.load(gan_dn_fp))
    if gan_gn_fp:
        gn.load_gn_state(tor.load(gan_gn_fp))

    # loss_func = tor.nn.CrossEntropyLoss().cuda()
    loss_func = tor.nn.BCELoss().cuda()

    # optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
    optim_gn = tor.optim.Adam(gn.parameters(), lr=LR)
    optim_dn = tor.optim.Adam(dn.parameters(), lr=LR)

    lr_step_gn = StepLR(optim_gn, step_size=LR_STEPSIZE, gamma=LR_GAMMA)
    lr_step_dn = StepLR(optim_dn, step_size=LR_STEPSIZE, gamma=LR_GAMMA)

    x = Variable(tor.FloatTensor(BATCHSIZE, LATENT_SPACE)).cuda()
    img = Variable(tor.FloatTensor(BATCHSIZE, 3, 64, 64)).cuda()

    dis_true = Variable(tor.ones(BATCHSIZE, 1)).cuda()
    dis_false = Variable(tor.zeros(BATCHSIZE, 1)).cuda()
    x_eval_train = Variable(x_eval_train).cuda()

    loss_real, loss_fake = None, None

    # ef bh(m, gi, go) :
    #   print (m)
    #   print (gi, go)
    # n.register_backward_hook(bh)
    # dn.register_backward_hook(bh)


    ### Training
    for epoch in range(EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1))

        for step, (x_batch, cls_batch) in enumerate(data_loader):
            print("Process: {}/{}".format(step, int(AVAILABLE_SIZE[0] / BATCHSIZE)), end="\r")

            ### train true/false pic
            if (step // PIVOT_STEPS) % 2 == 0:
                print("use dn")
                if step % 2 == 0:
                    img.data.copy_(x_batch)
                else:
                    rand_v = tor.FloatTensor(BATCHSIZE, LATENT_SPACE).uniform_(0, 1)
                    rand_v[:, 0] = tor.FloatTensor(BATCHSIZE).random_(0, 2)  # set attribute dim
                    x.data.copy_(rand_v)
                    out = gn(x)
                    img.data.copy_(out.data)

                dis = dis_true if step % 2 == 0 else dis_false
                cls = Variable(cls_batch).cuda()
                dis_pred, cls_pred = dn(img)
                optim = optim_dn

                loss_dis = loss_func(dis_pred, dis)
                loss_cls = loss_func(cls_pred, cls)
                loss = loss_dis + loss_cls if step % 2 == 0 else loss_dis

                if step % 2 == 0:
                    loss_real = loss_cls
                else:
                    loss_fake = loss_cls

            else:
                print("gn")
                rand_v = tor.FloatTensor(BATCHSIZE, LATENT_SPACE).uniform_(0, 1)
                rand_v[:, 0] = tor.FloatTensor(BATCHSIZE).random_(0, 2)  # set attribute dim
                x.data.copy_(rand_v)
                out = gn(x)
                dis = dis_true
                cls = Variable(cls_batch).cuda()
                dis_pred, cls_pred = dn(out)

                optim = optim_gn

                loss_dis = loss_func(dis_pred, dis)
                loss_cls = loss_func(cls_pred, cls)
                loss = (loss_dis + loss_cls)
                loss_fake = loss_cls
            loss.backward()

            optim.step()

            optim_dn.zero_grad()
            optim_gn.zero_grad()
            lr_step_dn.step()
            lr_step_gn.step()

            if step % RECORD_JSON_PERIOD == 0 and step != 0:
                x_true = x_eval_train
                dis, cls = dn(x_true)
                acc_true = round(int((dis > 0.5).sum().data) / EVAL_SIZE, 5)
                x_noise = tor.FloatTensor(EVAL_SIZE, 512).uniform_(0, 1)
                x_noise[:, 0] = tor.FloatTensor(EVAL_SIZE, 1).random_(0, 2)
                x_noise = Variable(x_noise).cuda()
                x_false = gn(x_noise)
                dis, cls = dn(x_false)
                acc_false = round(int((dis <= 0.5).sum().data) / EVAL_SIZE, 5)

                print("|Acc True: {}   |Acc False: {}".format(acc_true, acc_false))

                save_record(model_index, epoch, optim, loss_real, loss_fake, acc_true, acc_false)

            if step % RECORD_PIC_PERIOD == 0:
                loss = float(loss.data)
                print("|Loss: {:<8}".format(loss))
                save_pic_2("output_{}".format(model_index), gn, 4, epoch, step)


                ### Save model
            if step % RECORD_MODEL_PERIOD == 0:
                # tor.save(gn.state_dict(), os.path.join(MODEL_ROOT, "gan_gn_{}_{}.pkl".format(model_index, epoch)))
                # tor.save(dn.state_dict(), os.path.join(MODEL_ROOT, "gan_dn_{}_{}.pkl".format(model_index, epoch)))
                tor.save(gn.state_dict(), os.path.join(MODEL_ROOT, "gan_gn_{}.pkl".format(model_index, epoch)))
                tor.save(dn.state_dict(), os.path.join(MODEL_ROOT, "gan_dn_{}.pkl".format(model_index, epoch)))


""" Main """
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("-bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("-ps", action="store", type=int, default=None, help="pivot steps")
    parser.add_argument("--gn", action="store", type=str, default=None)
    parser.add_argument("--dn", action="store", type=str, default=None)
    parser.add_argument("--gangn", action="store", type=str, default=None)
    parser.add_argument("--gandn", action="store", type=str, default=None)

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i
    gn_fp = parser.parse_args().gn
    dn_fp = parser.parse_args().dn
    gan_gn_fp = parser.parse_args().gangn
    gan_dn_fp = parser.parse_args().gandn
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    BATCHSIZE = parser.parse_args().bs if parser.parse_args().bs else BATCHSIZE
    PIVOT_STEPS = parser.parse_args().ps if parser.parse_args().ps else PIVOT_STEPS

    ### Load Data
    console("Load Data")
    data_loader, x_eval_train = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index, x_eval_train, gn_fp, dn_fp, gan_gn_fp, gan_dn_fp)


