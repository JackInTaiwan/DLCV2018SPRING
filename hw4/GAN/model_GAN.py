import cv2
import torch as tor
import torch.nn as nn
import numpy as np
from torch.autograd import Variable




""" Module Build """
class GAN(nn.Module):
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
        super(GAN, self).__init__()
        self.index = 0
        self.training = True

        GN_conv_channels = [3, 2 ** 6, 2 ** 7, 2 ** 6, 2 ** 7, 2 ** 8, 3]
        GN_fc_channels = [2 ** 7 * 32 * 32, 2 ** 10, 2 ** 9, 2 ** 7, 3 * 2 ** 12]
        DN_conv_channels = [3, 2 ** 5, 2 ** 6, 2 ** 7]
        DN_fc_channels = [16 * 16 * DN_conv_channels[-1], 2 ** 10, 1]

        # Generator Network
        self.de_trans_1 = tor.nn.ConvTranspose2d(in_channels=fc_channels[2], out_channels=conv_channels[3], kernel_size=32, stride=1)
        self.de_conv_1 = self.conv(conv_channels[3], conv_channels[4], 3, 1)
        self.de_conv_2 = self.conv(conv_channels[4], conv_channels[5], 3, 1)
        self.de_trans_2 = tor.nn.ConvTranspose2d(in_channels=conv_channels[5], out_channels=conv_channels[6], kernel_size=2, stride=2, bias=False)
        self.out = tor.nn.Sigmoid()

        # Discriminator Network
        self.dn_conv_1 = self.conv(DN_conv_channels[0], DN_conv_channels[1], 3, 1)
        self.dn_pool_1 = tor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dn_conv_2 = self.conv(DN_conv_channels[1], DN_conv_channels[2], 3, 1)
        self.dn_conv_3 = self.conv(DN_conv_channels[2], DN_conv_channels[3], 3, 1)
        self.dn_pool_2 = tor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dn_fc_1 = self.fc(DN_fc_channels[0], DN_conv_channels[1])
        self.dn_fc_2 = self.fc(DN_fc_channels[1], DN_conv_channels[2])
        self.dn_sig = tor.nn.Sigmoid()


    def GN(self, x) :
        x = x.view(x.size(0), -1, 1, 1)
        x = self.de_trans_1(x)
        x = self.de_conv_1(x)
        x = self.de_conv_2(x)
        x = self.de_trans_2(x)
        out = self.out(x)

        return out


    def DN(self, x) :
        x = self.de_conv_1(x)
        x = self.de_pool_1(x)
        x = self.de_conv_2(x)
        x = self.de_conv_3(x)
        x = self.de_pool_2(x)
        x = x.view(x.size(0), -1)
        x = self.de_fc_1(x)
        x = self.de_fc_2(x)
        d = self.de_sig(x)

        return d


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
