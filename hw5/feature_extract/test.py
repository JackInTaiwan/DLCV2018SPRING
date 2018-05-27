import cv2
import random
import numpy as np
import torch as tor
from torch.autograd import Variable
from sklearn.model_selection import ShuffleSplit
#from model import Classifier


x = np.ones((1,5, 5, 3))
x = x.permute(0, 3, 1, 2)