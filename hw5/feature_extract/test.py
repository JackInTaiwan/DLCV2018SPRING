import cv2
import random
import numpy as np
import torch as tor
from sklearn.model_selection import ShuffleSplit
#from model import Classifier


x = [1,2,3]
y = [4,5,6]
p = (zip(x, y))
random.shuffle(p)
print (p)
x, y = zip(*p)
print (x, y)