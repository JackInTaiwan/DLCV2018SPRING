import cv2
import torchvision.models
import torch as tor
import torch.nn as nn
import numpy as np

from torch.autograd import Variable




class Classifier(nn.Module) :
    def __init__(self) :
        super(Classifier, self).__init__()

        self.index = 0
        self.lr = None
        self.lr_decay = None
        self.optim = None
        self.beta = None

        self.epoch = 1
        self.step = 1

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.eval()

        vgg16_fc_channels = [512 * 7 * 10, 2 ** 10]
        self.vgg16 = vgg16.features
        self.vgg16_fc_1 = nn.Linear(vgg16_fc_channels[0], vgg16_fc_channels[1])

        # output = (bs, 512, 7, 10)
        fc_channels = [vgg16_fc_channels[-1], 2 ** 11, 11]

        self.fc_1 = nn.Linear(fc_channels[0], fc_channels[1])
        self.fc_2 = nn.Linear(fc_channels[1], fc_channels[2])

        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid(inplace=True)


    def forward(self, x) :
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.vgg16_fc_1(x)
        return x


    def pred(self, x) :
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sig(x)
        return x


    def run_step(self) :
        self.step += 1


    def run_epoch(self) :
        self.epoch += 1


    def save(self, save_fp) :
        import torch as tor

        tor.save(self, save_fp)

        print ("===== Save Model =====")
        print ("|Model index: {}".format(self.index),
                "\n|Epoch: {}".format(self.epoch),
                "\n|Step: {}".format(self.step),
                "\n|Lr: {} |Lr_decay: {}".format(self.lr, self.lr_decay),
                "\n|Optim: {} |Beta: {}".format(self.optim, self.beta),
                "\n|Save path: {}".format(save_fp),
               )