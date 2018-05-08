import os
from argparse import ArgumentParser

import numpy as np
import torch as tor
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try :
    from model import AVE
    from utils import load_data, console, save_pic, record
except :
    from hw4.AVE.model import AVE
    from hw4.AVE.utils import load_data, console, save_pic, record




""" Parameters """
AVAILABLE_SIZE = None
EPOCH = 80
BATCHSIZE = 64
LR = 0.0001
LR_STEPSIZE = 700
LR_GAMMA = 0.95
MOMENTUM = 0.5
EVAL_SIZE = 100
RECORD_JSON_PERIOD = 10     # steps
RECORD_MODEL_PERIOD = 1     # epochs

KLD_LAMBDA = 10 ** -8

TRAIN_DATA_FP = ["./data/train_data.npy", "./data/train_data_1.npy", "./data/train_data_2.npy"]

RECORD_FP = "./record/model_ave.json"

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




def save_record(model_index, epoch, optim, recon_loss, KLD_loss) :
    record_data = dict()

    if epoch == 0:
        record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
        record_data["data_size"] = AVAILABLE_SIZE
        record_data["batch_size"] = BATCHSIZE
        record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
        record_data["lr_init"] = float(optim.param_groups[0]["lr"])
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["record_epoch"] = RECORD_MODEL_PERIOD
        record_data["recon_loss"] = round(float(recon_loss.data), 6)
        record_data["KLD_loss"] = round(float(KLD_loss), 6)

    else:
        record_data["model_name"] = "fcn_model_{}.pkl".format(model_index)
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["recon_loss"] = round(float(recon_loss.data), 6)
        record_data["KLD_loss"] = round(float(KLD_loss), 6)

    record(RECORD_FP, record_data)




""" Model Training """
def train(data_loader, model_index, x_eval_train, loaded_model):
    ### Model Initiation
    if loaded_model :
        ave = AVE()
        saved_state_dict = tor.load(loaded_model)
        ave.load_state_dict(saved_state_dict)
        ave.cuda()
    else :
        ave = AVE()
        ave.cuda()

    loss_func = tor.nn.MSELoss()

    #optim = tor.optim.SGD(fcn.parameters(), lr=LR, momentum=MOMENTUM)
    optim = tor.optim.Adam(ave.parameters(), lr=LR)

    lr_step = StepLR(optim, step_size=LR_STEPSIZE, gamma=LR_GAMMA)


    ### Training
    for epoch in range(46, EPOCH):
        print("|Epoch: {:>4} |".format(epoch + 1))

        ### Training
        for step, (x_batch, y_batch) in enumerate(data_loader):
            print ("Process: {}/{}".format(step, int(AVAILABLE_SIZE[0]/BATCHSIZE)), end="\r")
            x = Variable(x_batch).cuda()
            y = Variable(y_batch).cuda()
            out, KLD = ave(x)
            recon_loss = loss_func(out, y)
            loss = recon_loss + KLD_LAMBDA * KLD

            loss.backward()
            optim.step()
            lr_step.step()
            optim.zero_grad()

            if step % RECORD_JSON_PERIOD == 0 :
                save_record(model_index, epoch, optim, recon_loss, KLD)

        #print (out[:3])


        ### Evaluation
        loss = float(loss.data)
        print("|Loss: {:<8}".format(loss))


        ### Save output pictures
        save_pic("output", ave, 3)


        ### Save model
        if epoch % RECORD_MODEL_PERIOD == 0:
            tor.save(ave.state_dict(), os.path.join(MODEL_ROOT, "ave_model_{}_{}.pkl".format(model_index, epoch)))
        




""" Main """
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", action="store", type=int, default=False, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=False, help="amount of validation data")
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("--load", action="store", type=str, default=None)

    limit = parser.parse_args().l
    num_val = parser.parse_args().v
    model_index = parser.parse_args().i
    loaded_model = parser.parse_args().load
    LR = parser.parse_args().lr if parser.parse_args().lr else LR

    ### Load Data
    console("Load Data")
    data_loader, x_eval_train = data_loader(limit)

    ### Train Data
    console("Train Data")
    train(data_loader, model_index, x_eval_train, loaded_model)
