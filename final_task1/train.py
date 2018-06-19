import cv2
import random
import torch as tor
import numpy as np




""" Parameters """
CAL_ACC_PERIOD = 300    # steps
SHOW_LOSS_PERIOD = 30   # steps
SAVE_MODEL_PERIOD = 1   # epochs
SAVE_JSON_PERIOD = 50  # steps

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
VIDEOS_MAX_BATCH = 10
WAY = 20

EPOCH = 30
STEPS = 1000
BATCHSIZE = 1
LR = 0.0001
LR_STEPSIZE, LR_GAMMA = None, None




class Trainer :
    def __init__(self, recorder, base_train, novel_support, novel_test, shot, cpu=False) :
        self.recorder = recorder
        self.base_train = base_train
        self.novel_support = novel_support
        self.novel_test = novel_test
        self.shot = shot
        self.cpu = cpu


    def dump_novel_train(self) :
        way_pick = random.sample(range(self.base_train.shape[0]), WAY)
        shot_pick = random.sample(range(self.base_train.shape[1]), self.shot + 1)

        x = self.base_train[way_pick][:, shot_pick[:-1]]

        query_pick = random.choice(way_pick)
        x_query = self.base_train[query_pick, shot_pick[-1]][:]
        y_query = np.array(way_pick.index(query_pick))

        return  x, x_query, y_query


    def train(self) :
        model = self.recorder.models["matchnet"]
        if not self.cpu :
            model.cuda()

        # optim = tor.optim.SGD(model.fc_1.parameters(), lr=LR)
        optim = tor.optim.Adam(model.parameters(), lr=LR)
        loss_func = tor.nn.CrossEntropyLoss().cuda()

        for i in range(STEPS) :
            print("|Steps: {:>5} |".format(self.recorder.get_steps()))
            optim.zero_grad()

            x, x_query, y_query = self.dump_novel_train()
            print (x.shape)
            x = tor.Tensor(x)
            x_query = tor.Tensor(x_query)
            y_query = tor.Tensor(y_query)

            if not self.cpu :
                x, x_query, y_query = x.cuda(), x_query.cuda(), y_query.cuda()

            pred = model(x, x_query, y_query)

            loss = loss_func(pred, y_query)
            loss = float(loss.data)
            print("|Loss: {:<8}".format(loss))

            loss.backward()
            optim.step()


            self.recorder.step()


