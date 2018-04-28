import cv2
import os
import torch as tor
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try :
    from model import FCN
    from utils import load_data, console, record
except :
    from .model import FCN
    from .utils import load_data, console, record



""" Parameters """
AVAILABLA_SIZE = None
EPOCH = 50
BATCHSIZE = 16
LR = 0.0001
MOMENTUM = 0.5

EVAL_SIZE = 100
RECORD_MODEL_PERIOD = 10

X_TRAIN_FP = "./data/x_train.npy"
Y_TRAIN_FP = "./data/y_train.npy"

RECORD_FP = "./record/model_fcn.json"

MODEL_ROOT = "./models"



""" Data Setting """
def data_loader(limit) :
    x_train, y_train, x_size = load_data(X_TRAIN_FP, Y_TRAIN_FP)
    if limit :
        x_train, y_train = x_train[:limit], y_train[:limit]
    AVAILABLA_SIZE = str(x_train.shape)

    # Move axis in data for Pytorch
    x_train = np.moveaxis(x_train, 3, 1)
    y_train = y_train.astype(np.int16)
    x_train, y_train = tor.FloatTensor(x_train), tor.LongTensor(y_train)

    data_set = TensorDataset(
        data_tensor=x_train,
        target_tensor=y_train
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
    )


    return data_loader



""" Model Setup """
### Model Initiation
fcn = FCN()
fcn.vgg16_init()
fcn.cuda()

loss_func = tor.nn.CrossEntropyLoss()
optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
#optim = tor.optim.Adam(vgg.parameters(), lr=LR)
lr_schedule = StepLR(optim, step_size=1, gamma=0.9)



""" Model Training """
def train(data_loader, model_index) :
    for epoch in range(EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1), end="")

        ### Training
        lr_schedule.step()
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x = Variable(x_batch).type(tor.FloatTensor).cuda()
            y = Variable(y_batch).cuda()

            optim.zero_grad()
            pred = fcn(x)
            loss = loss_func(pred, y)
            loss.backward()
            optim.step()


        ### Evaluation
        #x_eval_train = Variable(x_train[:EVAL_SIZE]).type(tor.FloatTensor).cuda()
        #y_eval_train = Variable(y_train[:EVAL_SIZE]).type(tor.LongTensor).cuda()

        loss = loss_func(pred, y)
        acc = 0
        print ("Loss: {}".format(loss))


        ### Save model
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(fcn.state_dict(), os.path.join(MODEL_ROOT, "fcn_model_{}.pkl".format(model_index)))
            record_data = dict()
            if epoch == 0 :
                record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
                record_data["data_size"] = AVAILABLA_SIZE
                record_data["batch_size"] = BATCHSIZE
                record_data["lr_init"] = LR
                record_data["record_epoch"] = RECORD_MODEL_PERIOD
                record_data["lr"] = optim.param_groups[0]["lr"]
                record_data["loss"] = loss
                record_data["acc"] = acc

            else :
                record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
                record_data["lr"] = optim.param_groups[0]["lr"]
                record_data["loss"] = loss
                record_data["acc"] = acc

            record(RECORD_FP, record_data)



""" Main """
if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-i", action="store", type=int, required=True)

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i

    ### Load Data
    console("Load Data")
    data_loader = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index)
