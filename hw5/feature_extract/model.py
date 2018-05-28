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

        self.vgg16 = vgg16.features

        # output = (bs, 512, 7, 10)
        fc_channels = [512 * 7 * 10, 2 ** 10, 11]

        self.fc_1 = nn.Linear(fc_channels[0], fc_channels[1])
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(fc_channels[1], fc_channels[2])
        self.sig = nn.Sigmoid()


    def forward(self, x) :
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
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
        print ("|Model index: {} \n \
                |Epoch: {} \n \
                |Step: {} \n \
                |Lr: {} |Lr_decay: {} |\n \
                |Optim: {} |Beta:{} \n \
                |Save path: {}"
               .format(
                    self.index,
                    self.epoch,
                    self.step,
                    self.lr,
                    self.lr_decay,
                    self.optim,
                    self.beta,
                    save_fp,
                )
               )