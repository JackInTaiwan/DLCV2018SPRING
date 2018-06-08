import os
import threading
from argparse import ArgumentParser

import numpy as np
import torch as tor
from torch.optim.lr_scheduler import StepLR

from utils import console, accuracy, record, Batch_generator
from model import (
    RNN_1,
    RNN_2,
    RNN_3,
    RNN_4,
)




""" Parameters """
TRIMMED_LABEL_TRAIN_FP = ["./labels_train_0.npy", "./labels_train_1.npy", "./labels_train_2.npy", "./labels_train_3.npy"]
TRIMMED_VIDEO_TRAIN_FP = ["./videos_train_0.npy", "./videos_train_1.npy", "./videos_train_2.npy", "./videos_train_3.npy"]

TRIMMED_LABEL_VALID_FP = "./labels_valid_0.npy"
TRIMMED_VIDEO_VALID_FP = "./videos_valid_0.npy"

model_versions = [RNN_1, RNN_2, RNN_3, RNN_4]

RECORD_FP = "./record/"
MODEL_FP = "./models/"

CAL_ACC_PERIOD = 1000    # steps
SHOW_LOSS_PERIOD = 100   # steps
SAVE_MODEL_PERIOD = 1   # epochs
SAVE_JSON_PERIOD = 500  # steps

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
VIDEOS_MAX_BATCH = 10

EPOCH = 100
BATCHSIZE = 1
LR = 0.0001
LR_STEPSIZE, LR_GAMMA = 3000, 0.99

INPUT_SIZE, HIDDEN_SIZE= 1024, 512




""" Load Data """
def load(videos_fp, labels_fp, limit, val_limit) :
    videos = np.load(videos_fp)
    labels = np.load(labels_fp)

    if limit :
        videos = videos[:limit]
        labels = labels[:limit]

    if val_limit :
        videos_eval = videos[:val_limit][:]
        labels_eval = labels[:val_limit][:]

    else :
        videos_eval = videos[:EVAL_TRAIN_SIZE][:]
        labels_eval = labels[:EVAL_TRAIN_SIZE][:]

    videos_test = np.load(TRIMMED_VIDEO_VALID_FP)
    labels_test = np.load(TRIMMED_LABEL_VALID_FP)

    global AVAILABLE_SIZE
    AVAILABLE_SIZE = videos.shape[0]

    batch_gen = Batch_generator(
        x=videos,
        y=labels,
        batch=BATCHSIZE,
        drop_last=True,
    )

    return batch_gen, videos_eval, labels_eval, videos_test, labels_test




""" Save record """
def save_record(model_index, step, optim, loss, acc_train, acc_test):
    record_data = dict()

    model_name = "model_{}".format(model_index)

    if step == 1:
        record_data["model_name"] = model_name
        record_data["batch_size"] = BATCHSIZE
        record_data["decay"] = str((LR_STEPSIZE, LR_GAMMA))
        record_data["lr_init"] = float(optim.param_groups[0]["lr"])
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["record_period"] = SAVE_JSON_PERIOD

    else:
        record_data["model_name"] = model_name
        record_data["lr"] = float(optim.param_groups[0]["lr"])
        record_data["loss"] = round(loss, 6) if loss != None else None
        record_data["acc_train"] = acc_train
        record_data["acc_test"] = acc_test

    json_fp = os.path.join(RECORD_FP, "model_{}.json".format(model_index))

    if not os.path.exists(json_fp) :
        with open(json_fp, "w") as f :
            f.write('"[]"')

    thread_record = threading.Thread(target=record, args=[json_fp, record_data])
    thread_record.start()




""" Training """
def train(model, model_index, limit, valid_limit) :
    epoch_start = model.epoch

    optim = tor.optim.Adam(model.parameters(), lr=LR)

    lr_schedule = StepLR(optimizer=optim, step_size=LR_STEPSIZE, gamma=LR_GAMMA)

    loss_func = tor.nn.CrossEntropyLoss().cuda()

    loss_total = np.array([])

    for epoch in range(epoch_start, epoch_start + EPOCH) :
        print("|Epoch: {:>4} |".format(epoch))

        for videos_fp, labels_fp in zip(TRIMMED_VIDEO_TRAIN_FP, TRIMMED_LABEL_TRAIN_FP) :
            batch_gen, x_eval_train, y_eval_train, x_eval_test, y_eval_test = load(videos_fp, labels_fp, limit, valid_limit)

            for (x_batch, y_batch) in batch_gen:
                step = model.step
                print("Process: {}/{}".format(step % int(AVAILABLE_SIZE / BATCHSIZE) , int(AVAILABLE_SIZE / BATCHSIZE)), end="\r")
                x = tor.FloatTensor(x_batch).cuda()
                y = tor.LongTensor(y_batch).cuda()

                optim.zero_grad()

                pred = model(x)

                loss = loss_func(pred, y)
                loss_total = np.concatenate((loss_total, [loss.data.cpu().numpy()]))

                loss.backward()
                optim.step()
                lr_schedule.step()
                model.run_step()

                if step == 1 :
                    save_record(model_index, step, optim, None, None, None)

                if step % SHOW_LOSS_PERIOD == 0 :
                    print("|Loss: {}".format(loss_total.mean()))
                    save_record(model_index, step, optim, loss_total.mean(), None, None)
                    loss_total = np.array([])

                if step % CAL_ACC_PERIOD == 0 :
                    model.eval()

                    acc_train = accuracy(model, x_eval_train, y_eval_train)
                    acc_test = accuracy(model, x_eval_test, y_eval_test)

                    model.train()

                    save_record(model_index, step, optim, None, acc_train, acc_test)

                    print ("|Acc on train data: {}".format(round(acc_train, 5)))
                    print ("|Acc on test data: {}".format(round(acc_test, 5)))

            if epoch % SAVE_MODEL_PERIOD == 0:
                save_model_fp = os.path.join(MODEL_FP, "model_{}.pkl".format(model_index))
                model.save(save_model_fp)

        model.run_epoch()





""" Main """
if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=None, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=None, help="amount of validation data")
    parser.add_argument("-e", action="store", type=int, help="epoch")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning reate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--load", action="store", type=str, help="file path of loaded model")
    parser.add_argument("--version", action="store", type=int, default=0, help="version of model")

    limit = parser.parse_args().l
    valid_limit = parser.parse_args().v
    model_index = parser.parse_args().i
    load_model_fp = parser.parse_args().load
    cpu = parser.parse_args().cpu
    model_version = parser.parse_args().version
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    BATCHSIZE = parser.parse_args().bs if parser.parse_args().bs else BATCHSIZE
    EPOCH = parser.parse_args().e if parser.parse_args().e else EPOCH


    ### Building Model
    console("Building Model")
    if load_model_fp :
        model = tor.load(load_model_fp)
    else :
        Model = model_versions[model_version] if model_version == 0 else model_versions[model_version - 1]
        model = Model(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
        )

    if not cpu :
        model.cuda()


    ### Train Data
    console("Training Model")
    train(model, model_index, limit, valid_limit)
