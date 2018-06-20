import cv2
import random
import torch as tor
import numpy as np




""" Parameters """
CAL_ACC_PERIOD = 500    # steps
SHOW_LOSS_PERIOD = 10   # steps
SAVE_MODEL_PERIOD = 1000   # epochs
SAVE_JSON_PERIOD = 50  # steps

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
EVAL_TEST_SIZE = 25

EPOCH = 30
STEPS = 100000
BATCHSIZE = 1
LR = 0.0001
LR_STEPSIZE, LR_GAMMA = None, None




class Trainer :
    def __init__(self, recorder, base_train, novel_support, novel_test, shot, way, cpu=False, lr=LR) :
        self.recorder = recorder
        self.base_train = base_train
        self.novel_support = novel_support
        self.novel_test = novel_test
        self.way = way
        self.shot = shot
        self.cpu = cpu

        self.model = self.recorder.models["relationnet"]
        self.model.way, self.model.shot = self.way, self.shot
        if not self.cpu :
            self.model.cuda()

        # optim = tor.optim.SGD(model.fc_1.parameters(), lr=LR)
        self.optim = tor.optim.Adam(self.model.parameters(), lr=LR)
        #self.loss_func = tor.nn.CrossEntropyLoss().cuda()
        self.loss_fn = tor.nn.MSELoss().cuda()



    def dump_novel_train(self) :
        way_pick = random.sample(range(self.base_train.shape[0]), self.way)
        shot_pick = random.sample(range(self.base_train.shape[1]), self.shot + 1)

        x = self.base_train[way_pick][:, shot_pick[:-1]]

        query_pick = random.choice(way_pick)
        x_query = self.base_train[query_pick, shot_pick[-1]]
        y_query = np.array([way_pick.index(query_pick)])

        return  x, x_query, y_query



    def eval(self) :
        self.model.way = 20
        self.model.shot = 5
        self.novel_support_tr = tor.Tensor(self.novel_support).cuda()
        correct, total = 0, self.novel_test.shape[0] * self.novel_test.shape[1]

        for label_idx, data in enumerate(self.novel_test) :
            for img in data[:EVAL_TEST_SIZE] :
                img = tor.Tensor(img).unsqueeze(0).cuda()
                scores = self.model(self.novel_support_tr, img)
                pred = int(tor.argmax(scores))
                print (pred, label_idx)
                if pred == label_idx :
                    correct += 1

        self.model.way = self.way
        self.model.shot = self.shot

        return correct / total



    def train(self) :

        for i in range(STEPS) :
            print("|Steps: {:>5} |".format(self.recorder.get_steps()), end="\r")
            self.optim.zero_grad()

            x, x_query, y_query_idx = self.dump_novel_train()
            x = tor.Tensor(x)
            x_query = tor.Tensor(x_query).unsqueeze(0)
            y_query = tor.zeros(self.way, 1)
            y_query[y_query_idx] = 1

            if not self.cpu :
                x, x_query, y_query = x.cuda(), x_query.cuda(), y_query.cuda()

            scores = self.model(x, x_query)

            loss = self.loss_fn(scores, y_query)
            loss.backward()


            if self.recorder.get_steps() % SHOW_LOSS_PERIOD == 0 :
                print("|Loss: {:<8}".format(float(loss.data)))

            if self.recorder.get_steps() % SAVE_JSON_PERIOD == 0 :
                self.recorder.save_checkpoints()

            if self.recorder.get_steps() % SAVE_MODEL_PERIOD == 0 :
                self.recorder.save_models()

            if self.recorder.get_steps() % CAL_ACC_PERIOD == 0 :
                acc = self.eval()
                print("|Acc: {:<8}".format(round(acc, 5)))

            self.optim.step()
            self.recorder.step()


