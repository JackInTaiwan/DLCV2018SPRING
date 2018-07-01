import random
import torch as tor
import numpy as np
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset




""" Parameters """
CAL_ACC_PERIOD = 1000  # steps
SHOW_LOSS_PERIOD = 100  # steps
SAVE_MODEL_PERIOD = 1000  # epochs
SAVE_JSON_PERIOD = 50  # steps

AVAILABLE_SIZE = None
EVAL_TRAIN_SIZE = 100
EVAL_TEST_SIZE = 50
EVAL_NOVEL_SIZE = 200


EPOCH = 30
STEP = 50000
BATCHSIZE = 32
LR = 0.00001
LR_STEPSIZE, LR_GAMMA = 20000, 0.95
LDA = 1.0




class Trainer:
    def __init__(self, recorder, base_train, base_test, novel_support, novel_test, shot, way, cpu=False, lr=LR, step=None):
        self.recorder = recorder
        self.base_train = base_train
        self.base_test = base_test
        self.novel_support = novel_support
        self.novel_test = novel_test
        self.way = way
        self.shot = shot
        self.cpu = cpu
        self.lr = lr
        self.step = step if step else STEP

        self.model = self.recorder.models["classifier"]
        if not self.cpu:
            self.model.cuda()

        # self.optim = tor.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optim = tor.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = tor.nn.CrossEntropyLoss().cuda()
        #self.loss_fn = tor.nn.MSELoss().cuda()
        self.lr_schedule = tor.optim.lr_scheduler.StepLR(optimizer=self.optim, step_size=LR_STEPSIZE, gamma=LR_GAMMA)


    def eval_novel(self):
        self.model.eval()

        novel_support = tor.Tensor(self.novel_support).permute(0, 1, 4, 2, 3).cuda()
        novel_test = tor.Tensor(self.novel_test[:, :EVAL_NOVEL_SIZE]).permute(0, 1, 4, 2, 3).cuda()

        pred = self.model.pred(novel_support, novel_test)
        labels = np.array([j // EVAL_NOVEL_SIZE for j in range(EVAL_NOVEL_SIZE * 20)])
        acc = np.mean(pred == labels)

        self.model.train()
        novel_support = novel_support.cpu()
        novel_support = novel_test.cpu()

        return acc



    def eval_test(self) :
        x = self.base_test[:, :EVAL_TEST_SIZE].reshape(-1, 32, 32, 3)
        y = np.array([i // EVAL_TEST_SIZE for i in range(80 * EVAL_TEST_SIZE)])
        x, y = tor.Tensor(x), tor.tensor(y, dtype=tor.long)

        data_set = TensorDataset(x, y)

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=BATCHSIZE,
            shuffle=False,
            drop_last=True,
        )

        self.model.eval()
        acc_list = []
        for x, y in data_loader :
            x, y = x.permute(0, 3, 1, 2).cuda(), y.cuda()
            pred = self.model(x)
            acc = np.mean((tor.argmax(pred, dim=1).view(-1) == y.view(-1)).cpu().detach().numpy())
            acc_list.append(acc)
        self.model.train()

        return float(np.mean(np.array(acc_list)))


    def dump_data(self) :
        label_pick = random.sample(range(80), BATCHSIZE)
        x_1 = self.base_train[label_pick]
        x_1 = x_1[:, random.randrange(500)]
        x_2 = self.base_train[label_pick][:, random.randrange(500)]
        x = np.vstack((x_1, x_2))
        y = np.array(label_pick * 2)

        return x, y


    def train(self) :
        self.model.train()

        loss_list = []
        train_acc_list = []

        while self.step > 0 :
            x, y = self.dump_data()
            print("|Steps: {:>5} |".format(self.recorder.get_steps()), end="\r")
            self.optim.zero_grad()

            x = x.permute(0, 3, 1, 2)
            print (x.size())
            if not self.cpu:
                x, y = x.cuda(), y.cuda()

            scores, features = self.model(x)

            # calculate training accuracy
            acc = (tor.argmax(scores, dim=1) == y.view(-1).cuda())
            acc = np.mean(acc.cpu().numpy())
            train_acc_list.append(acc)

            loss_cls = self.loss_func(scores, y)
            loss_sim = (cosine_similarity(features[0], features[1], dim=1) * -1 + 1.0) / features[0].size(0)
            loss = loss_cls + LDA * loss_sim
            loss.backward()

            loss_list.append(float(loss.data))

            if self.recorder.get_steps() % SHOW_LOSS_PERIOD == 0:
                loss_avg = round(float(np.mean(np.array(loss_list))), 6)
                train_acc_avg = round(float(np.mean(np.array(train_acc_list))), 5)
                self.recorder.checkpoint({
                    "loss": loss_avg,
                    "train_acc": train_acc_avg
                })
                print("|Loss: {:<8} |Train Acc: {:<8}".format(loss_avg, train_acc_avg))

                loss_list = []
                train_acc_list = []

            if self.recorder.get_steps() % SAVE_JSON_PERIOD == 0:
                self.recorder.save_checkpoints()

            if self.recorder.get_steps() % SAVE_MODEL_PERIOD == 0:
                self.recorder.save_models()

            if self.recorder.get_steps() % CAL_ACC_PERIOD == 0:
                acc = self.eval_novel()
                acc_test = self.eval_test()
                self.recorder.checkpoint({
                    "acc": acc,
                    "lr": self.optim.param_groups[0]["lr"]
                })
                print("|Novel Acc: {:<8} | Test Acc: {:<8}".format(round(acc, 5), round(acc_test, 5)))

            self.optim.step()
            self.lr_schedule.step()
            self.recorder.step()
            self.step -= 1
