import cv2
import numpy as np
import torch as tor

from torch.autograd import Variable
from model import AVE

ave = AVE()
for item in ave.parameters() :
    print (item)