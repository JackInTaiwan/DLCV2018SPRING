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
    correct, total = 0, len(labels[0])

    for i, (x, label) in enumerate(zip(data[0], labels[0]), 1) :
        print ("Accuracy Process: {}/{}".format(i, total), end="\r")

        x = tor.Tensor(x).unsqueeze(0).unsqueeze(0).cuda()

        o, h = model(x) if i == 1 else model(x, h[0].detach(), h[1].detach())

        pred = tor.max(o, 1)[1]

        if int(pred[0].data) == label :
            correct += 1

        if int(pred[0].data) != 0 : print ("!!!!!!!!!!!!!!!!")
    
    return correct / total
