import cv2
import random
import numpy as np
import torch as tor
from torch.autograd import Variable
from sklearn.model_selection import ShuffleSplit
#from model import Classifier


x = Variable(tor.FloatTensor(tor.ones((3, 10))))
print (x.mean(dim=0).unsqueeze(0))