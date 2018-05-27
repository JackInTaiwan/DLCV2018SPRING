import math
import random
import numpy as np




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

        if self.index < max_index :
            self.index += 1
            return np.array(self.x[self.batch * self.index: self.batch * (self.index + 1)]), np.array(self.y[self.batch * self.index: self.batch * (self.index + 1)])

        else :
            self.index = 0
            raise StopIteration