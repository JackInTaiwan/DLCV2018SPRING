import cv2
import math
import random
import numpy as np
import torch as tor

from torch.autograd import Variable




class Batch_generator() :

    def __init__(self, x, y, batch, drop_last=True) :
        self.x = x
        self.y = y
        self.batch = batch
        self.drop_last = drop_last
        self.index = 0


    def __iter__(self) :
        pairs = list(zip(self.x, self.y))
        random.shuffle(pairs)
        self.x, self.y = zip(*pairs)
        return self


    def __next__(self) :
        max_index = len(self.x) // self.batch if self.drop_last else math.ceil(len(self.x)/self.batch)

        if self.index < max_index - 1 :
            self.index += 1
            return np.array(self.x[self.batch * self.index: self.batch * (self.index + 1)]), np.array(self.y[self.batch * self.index: self.batch * (self.index + 1)])

        else :
            self.index = 0
            raise StopIteration




def accuracy(model, data, labels) :
    acc_list = []
    for d in range(len(data)) :
        correct, total = 0, len(labels[d])

        x = tor.Tensor(data[d]).unsqueeze(0).cuda()
        o, h = model(x)
        print (o[0])
        pred = tor.max(o, 1)[1]
        y = tor.LongTensor(labels[d].astype(np.uint8)).cuda()
        correct = tor.sum(y == pred)
        correct = int(correct.data.cpu())
        zeros = int(tor.sum(pred == 0).data.cpu())
        acc_list.append((correct / total, (zeros, total)))

    return acc_list
