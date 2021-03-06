import cv2
import os
import torch as tor
import numpy as np
import time
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try :
    from model import FCN
    from utils import load_data, console, record, evaluate
except :
    from .model import FCN
    from .utils import load_data, console, record, evaluate




""" Parameters """
AVAILABLA_SIZE = None
EPOCH = 5
BATCHSIZE = 15
LR = 0.0001
LR_STEPSIZE = 1
LR_GAMMA = 0.95
MOMENTUM = 0.5
EVAL_SIZE = 100
RECORD_MODEL_PERIOD = 1

X_TRAIN_FP = "./data/x_train.npy"
Y_TRAIN_FP = "./data/y_train.npy"

RECORD_FP = "./record/model_fcn.json"

MODEL_ROOT = "./models"



""" Convert labels """
def convert_labels(y) :
    y_o = np.zeros((y.shape[0], 7, y.shape[1], y.shape[2]))
    for i, layer in enumerate(y):
        for p in range(layer.shape[0]):
            for q in range(layer.shape[1]):
                y_o[i][int(layer[p][q])][p][q] = 1
    return y_o


""" Data Setting """
def data_loader(limit) :
    x_train, y_train, x_size = load_data(X_TRAIN_FP, Y_TRAIN_FP)
    print (x_train.shape)
    if limit :
        x_train, y_train = x_train[:limit], y_train[:limit]
    #y_train = convert_labels(y_train)

    global AVAILABLA_SIZE
    AVAILABLA_SIZE = str(x_train.shape)

    # Move axis in data for Pytorch
    #x_train = np.moveaxis(x_train, 3, 1)
    y_train = y_train.astype(np.int16)

    x_train, y_train = tor.FloatTensor(x_train).permute(0,3,2,1), tor.FloatTensor(y_train)
    
    x_eval_train, y_eval_train = x_train[:EVAL_SIZE], y_train[:EVAL_SIZE]

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


    return data_loader, x_eval_train, y_eval_train




""" Model Training """
def train(data_loader, model_index, x_eval_train, y_eval_train) :
    ### Model Initiation
    fcn = FCN()
    #print (fcn.b_1_conv_1[0].weight.data)

    d = tor.load("./models/vgg16_pretrained.pkl")
    fcn.vgg16_load(d)
    #d = tor.load("./models/fcn_model_1_1.pkl")
    #fcn.load_state_dict(d)
    fcn.cuda()
    #loss_func = tor.nn.CrossEntropyLoss(weight=w)
    loss_func = tor.nn.CrossEntropyLoss()
    #loss_func = tor.nn.MSELoss()
    #optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
    optim1 = tor.optim.Adam(fcn.b_6_conv_1.parameters(), lr=LR) 
    
    optim2 = tor.optim.Adam(fcn.b_6_conv_2.parameters(), lr=LR)
    optim3 = tor.optim.Adam(fcn.b_6_conv_3.parameters(), lr=LR) 
    optim4 = tor.optim.Adam(fcn.b_7_trans_1.parameters(), lr=LR)
    optim = tor.optim.Adam(fcn.parameters(), lr=LR)
    ### Training
    for epoch in range(EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1), end="")

        ### Training
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x = Variable(x_batch).type(tor.FloatTensor).cuda()
            y = Variable(y_batch).type(tor.LongTensor).cuda()
            
            pred = fcn(x)
            optim1.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()
            optim4.zero_grad()
            optim.zero_grad()
            loss = loss_func(pred, y)
            loss.backward()
            optim1.step()
            optim2.step()
            optim3.step()
            optim4.step()
        print (pred[:2])
        print (tor.max(pred[:5], 1)[1])
        ### Evaluation
        loss = float(loss.data)      
        acc = evaluate(fcn, x_eval_train, y_eval_train)

        print ("|Loss: {:<8} |Acc: {:<8}".format(loss, acc))


        ### Save model
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(fcn.state_dict(), os.path.join(MODEL_ROOT, "fcn_model_{}_{}.pkl".format(model_index, epoch)))


        ### Record
        record_data = dict()
        if epoch == 0 :
            record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
            record_data["data_size"] = AVAILABLA_SIZE
            record_data["batch_size"] = BATCHSIZE
            record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
            record_data["lr_init"] = float(optim1.param_groups[0]["lr"])
            record_data["lr"] = float(optim1.param_groups[0]["lr"])
            record_data["record_epoch"] = RECORD_MODEL_PERIOD
            record_data["loss"] = loss
            record_data["acc"] = acc
        else :
            record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
            record_data["lr"] = float(optim1.param_groups[0]["lr"])
            record_data["loss"] = loss
            record_data["acc"] = acc

        record(RECORD_FP, record_data)
        
 


""" Main """
if __name__ == "__main__" :
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
    data_loader, x_eval_train, y_eval_train = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index, x_eval_train, y_eval_train)
