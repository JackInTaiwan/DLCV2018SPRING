import os
import cv2
import numpy as np
import torch as tor

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try :
    from model import AVE
    from utils import load_data, console, save_pic
except :
    from .model import AVE
    from .utils import load_data, console, save_pic




""" Parameters """
AVAILABLA_SIZE = None
EPOCH = 10
BATCHSIZE = 64
LR = 0.0001
LR_STEPSIZE = 1
LR_GAMMA = 0.1
MOMENTUM = 0.5
EVAL_SIZE = 100
RECORD_MODEL_PERIOD = 1

KLD_LAMBDA = 10 ** -5

TRAIN_DATA_FP = "./data/train_data.npy"

RECORD_FP = "./record/model_fcn.json"

MODEL_ROOT = "./models"




""" Data Setting """
def data_loader(limit):
    x_train, x_size = load_data(TRAIN_DATA_FP)
    print(x_train.shape)

    if limit:
        x_train = x_train[:limit]

    global AVAILABLA_SIZE
    AVAILABLA_SIZE = str(x_train.shape)

    x_train /= 255.
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




""" Model Training """
def train(data_loader, model_index, x_eval_train):
    ### Model Initiation
    ave = AVE()
    ave.cuda()

    loss_func = tor.nn.MSELoss()

    # optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
    optim = tor.optim.Adam(ave.parameters(), lr=LR)

    lr_step = StepLR(optim, step_size=LR_STEPSIZE, gamma=LR_GAMMA)

    ### Training
    for epoch in range(EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1), end="")

        ### Training
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x = Variable(x_batch).cuda()
            y = Variable(y_batch).cuda()

            out, KLD = ave(x)

            loss = loss_func(out, y) +  KLD_LAMBDA * KLD

            loss.backward()
            optim.step()
            lr_step.step()
            optim.zero_grad()

        ### Evaluation
        loss = float(loss.data)
        #acc = evaluate(fcn, x_eval_train, y_eval_train)

        #print("|Loss: {:<8} |Acc: {:<8}".format(loss, acc))
        print("|Loss: {:<8}".format(loss))


        ### Save output pictures
        save_pic("output", ave, 3)


        ### Save model
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(ave.state_dict(), os.path.join(MODEL_ROOT, "fcn_model_{}_{}.pkl".format(model_index, epoch)))
        
        ### Record
        """
        record_data = dict()
        if epoch == 0:
            record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
            record_data["data_size"] = AVAILABLA_SIZE
            record_data["batch_size"] = BATCHSIZE
            record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
            record_data["lr_init"] = float(optim.param_groups[0]["lr"])
            record_data["lr"] = float(optim.param_groups[0]["lr"])
            record_data["record_epoch"] = RECORD_MODEL_PERIOD
            record_data["loss"] = loss
            record_data["acc"] = acc
        else:
            record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
            record_data["lr"] = float(optim1.param_groups[0]["lr"])
            record_data["loss"] = loss
            record_data["acc"] = acc

        record(RECORD_FP, record_data)
        """


""" Main """
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-lr", action="store", type=float, default=False, help="learning reate")

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i
    LR = parser.parse_args().lr if parser.parse_args().lr else LR

    ### Load Data
    console("Load Data")
    data_loader, x_eval_train = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index, x_eval_train)
