import cv2
import random
import numpy as np
import torch as tor
from torch.autograd import Variable
from sklearn.model_selection import ShuffleSplit
#from model import Classifier


x = Variable(tor.FloatTensor(np.array([1,2,3])))
y = tor.max(x, 0)[1]
print (y.data)