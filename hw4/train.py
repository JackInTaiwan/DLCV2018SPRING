import cv2
import numpy as np
import torch as tor

from torch.autograd import Variable
from model_test import AVE


data = np.load("./data/train_data.npy")
x = data[:10]
x = np.array(x)
x = Variable(tor.FloatTensor(x))
print (x.size())
ave = AVE()

y = ave(x)
print (y)
