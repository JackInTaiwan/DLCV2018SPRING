import cv2
import os
from argparse import ArgumentParser

import numpy as np
import torch as tor
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from model import AVE
    from utils import load_data, console, save_pic, record
except:
    from hw4.AVE.model import AVE
    from hw4.AVE.utils import load_data, console, save_pic, record
