import cv2
import torch as tor
import h5py
import numpy as np
from torch.autograd import Variable

try :
    from model import FCN
except :
    from .model import FCN



x_train = np.load("./data/x_train.npy")

fcn = FCN()
fcn.vgg16_init()
x_v = Variable(tor.FloatTensor(np.array([x_train])))
pred = fcn(x_v)
print (pred)
