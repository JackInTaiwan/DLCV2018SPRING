import cv2
import torch as tor
import torch.nn as nn
import numpy as np



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
            nn.ReLU(),
        )
        return conv


    def fc(self, num_in, num_out) :
        fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
        )
        return fc


    def __init__(self):
        super(AVE, self).__init__()
        self.index = 0

        conv_channels = np.array([3, 2 ** 8, 2 ** 9])
        conv_channels = [int(num) for num in conv_channels]    # transform type
        fc_channels = np.array([2 ** 9 * 32 * 32, 2 ** 9, 2 ** 9, 2 ** 10, 3 * 2 ** 12])
        fc_channels = [int(num) for num in fc_channels]  # transform type

        # block 1
        self.en_conv_1 = self.conv(conv_channels[0], conv_channels[1], 3, 1)
        self.en_conv_2 = self.conv(conv_channels[1], conv_channels[2], 3, 1)
        self.en_pool_1 = tor.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.en_fc_1 = self.fc(fc_channels[0], fc_channels[1])

        # latent space
        self.ls_fc_1 = self.fc(fc_channels[1], fc_channels[2])

        # logvar
        self.lv_fc_1 = self.fc(fc_channels[1], fc_channels[2])

        # decode
        self.de_fc_1 = self.fc(fc_channels[2], fc_channels[3])
        self.de_fc_2 = self.fc(fc_channels[3], fc_channels[4])
        self.out = tor.nn.Sigmoid()


    def encode(self, x) :
        x = self.en_conv_1(x)
        x = self.en_conv_2(x)
        x = self.en_pool_1(x)
        x = self.en_fc_1(x)

        ls = self.ls_fc_1(x)
        logvar = self.lv_fc_1(x)

        KLD = -0.5 * tor.sum(1 + logvar - ls.pow(2) - logvar.exp())

        return ls, logvar, KLD


    def decode(self, ls, logvar, training=True) :
        if training :
            x = ls + tor.randn() * (logvar * 0.5).exp()
        else :
            x = ls

        x = self.de_fc_1(x)
        x = self.de_fc_2(x)
        out = self.out(x)

        out = out.view(-1, 3, 64, 64)

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
