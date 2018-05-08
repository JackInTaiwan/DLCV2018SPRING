import cv2
import torch as tor
import torch.nn as nn
import numpy as np
from torch.autograd import Variable




""" Parameters """
VGG16_PRETRAINED_FP = "./models/vgg16_pretrained.h5"



""" Module Build """
class AVE(nn.Module):
    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_conv_channels,
                out_channels=out_conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
            nn.ReLU(inplace=True),
        )
        return conv


    def fc(self, num_in, num_out) :
        fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(inplace=True)
        )
        return fc


    def __init__(self):
        super(AVE, self).__init__()
        self.index = 0
        self.training = True

        conv_channels = np.array([3, 2 ** 6, 2 ** 9, 2 ** 10, 3, 2 ** 8, 3])
        conv_channels = [int(num) for num in conv_channels]    # transform type
        fc_channels = np.array([2 ** 7 * 32 * 32, 2 ** 10, 2 ** 9, 2 ** 7, 3 * 2 ** 12])
        fc_channels = [int(num) for num in fc_channels]  # transform type

        # block 1
        self.en_conv_1 = self.conv(conv_channels[0], conv_channels[1], 3, 1)
        self.en_conv_2 = self.conv(conv_channels[1], conv_channels[2], 3, 1)
        self.en_pool_1 = tor.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.en_fc_1 = self.fc(fc_channels[0], fc_channels[1])

        # latent space
        self.ls_fc_1 = self.fc(fc_channels[1], fc_channels[2])
        self.ls_tanh = tor.nn.Tanh()
        self.ls_sig = tor.nn.Sigmoid()

        # logvar
        self.lv_fc_1 = self.fc(fc_channels[1], fc_channels[2])
        self.lv_tanh = tor.nn.Tanh()
        self.lv_sig = tor.nn.Sigmoid()

        # decode
        self.de_trans_1 = tor.nn.ConvTranspose2d(in_channels=fc_channels[2], out_channels=conv_channels[3], kernel_size=32, stride=1)
        #self.de_conv_1 = self.conv(conv_channels[3], conv_channels[4], 3, 1)
        #self.de_conv_2 = self.conv(conv_channels[4], conv_channels[5], 3, 1)
        self.de_trans_2 = tor.nn.ConvTranspose2d(in_channels=conv_channels[3], out_channels=conv_channels[4], kernel_size=2, stride=2, bias=False)
        self.out = tor.nn.Sigmoid()


    def encode(self, x) :
        x = self.en_conv_1(x)
        x = self.en_conv_2(x)
        x = self.en_pool_1(x)
        x = x.view(x.size(0), -1)
        x = self.en_fc_1(x)

        ls = self.ls_tanh(self.ls_fc_1(x))
        logvar = self.lv_tanh(self.lv_fc_1(x))
        #ls = self.ls_sig(self.ls_fc_1(x))
        #logvar = self.lv_sig(self.lv_fc_1(x))

        KLD = -0.5 * tor.sum(1 + logvar - ls.pow(2) - logvar.exp())

        return ls, logvar, KLD


    def decode(self, ls, logvar) :
        if self.training :
            ex = (logvar * 0.5).exp()
            noise = Variable(tor.randn(tuple(logvar.size()))).cuda()
            x = ls + noise.mul(ex)
        else :
            x = ls
        x = x.view(x.size(0), -1, 1, 1)
        x = self.de_trans_1(x)
        x = self.de_conv_1(x)
        x = self.de_conv_2(x)
        x = self.de_trans_2(x)
        out = self.out(x)

        return out


    def forward(self, x):
        ls, logvar, KLD = self.encode(x)

        output = self.decode(ls, logvar)

        return output, KLD


    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.001)
            tor.nn.init.normal(m.bias, 0, 0.001)
        elif classname.find("Conv") != -1 and self.index >=44 :
            m.weight.data.normal_(0.01, 0.001)
            m.bias.data.normal_(0.01, 0.001)
        self.index += 1
