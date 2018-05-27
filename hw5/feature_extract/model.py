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
        self.fc_1 = nn.Linear(512 * 7 * 10, 11)
        self.sig = nn.Sigmoid()


    def forward(self, x) :
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.sig(x)
        return x


    def cls(self, x) :
        x = self.fc_1(x)
        x = self.sig(x)
        return x


    def step(self) :
        self.step += 1


    def epoch(self) :
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