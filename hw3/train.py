import cv2
import torch as tor
import numpy as np
import time
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try :
    from model import FCN
    from utils import load_data, console
except :
    from .model import FCN
    from .utils import load_data, console



AVAILABLA_SIZE = None
EPOCH = 10
BATCHSIZE = 8
LR = 0.001
MOMENTUM = 0.5

EVAL_SIZE = 100
RECORD_MODEL_PERIOD = 10

X_TRAIN_FP = "./data/x_train.npy"
Y_TRAIN_FP = "./data/y_train.npy"



""" Data Setting """
def data_loader(limit) :
    x_train, y_train, x_size = load_data(X_TRAIN_FP, Y_TRAIN_FP)
    if limit :
        x_train, y_train = x_train[:limit], y_train[:limit]

    # Move axis in data for Pytorch
    x_train = np.moveaxis(x_train, 3, 1)
    s = time.time()
    y_train = y_train.astype(np.int16)
    e = time.time()
    print ("transfer y time", e-s)
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
lr_schedule = StepLR(optim, step_size=20, gamma=0.9)



""" Model Training """
def train(data_loader) :
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
        print (pred[0])
        print (y[0])
        loss = loss_func(pred, y)
        print (loss)
        #loss, acc = evaluation(vgg, loss_func, x_eval_train, y_eval_train)
        #print("Acc: {:<7} |Loss: {:<7} |".format(acc, loss))


        ### Save model
        """
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(vgg.state_dict(), MODEL_ROOT + "vgg16_model_{}.pkl".format(model_index))
    
            output(fp_record,
                   "|Epoch: {:<4} |LR: {:<7} |Acc: {:<7} |Loss: {:<7} |".format(epoch + 1, optim.param_groups[0]["lr"],
                                                                                acc, loss))
            t = time.localtime()
            output(fp_record,
                   "Saving Time: {:<4}/{:<2}/{:<2} {:<2}:{:<2}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                                                                       t.tm_min))
            output(fp_record, "\n")
        """



""" Main """
if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")

    limit = parser.parse_args().l
    num_val = parser.parse_args().v

    ### Load Data
    console("Load Data")
    data_loader = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader)
