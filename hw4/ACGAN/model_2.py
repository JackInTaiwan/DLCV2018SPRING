import cv2
import torch as tor
import torch.nn as nn
import numpy as np
from torch.autograd import Variable




""" Module Build """
class GN(nn.Module) :
    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride, relu=True):
        conv_relu = nn.Sequential(
            nn.Conv2d(
                in_channels=in_conv_channels,
                out_channels=out_conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
            nn.ReLU(inplace=True),
        )
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_conv_channels,
                out_channels=out_conv_channels,
                kernel_size=kernel_size, 
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
        )
        return conv_relu if relu else conv


    def __init__(self):
        super(GN, self).__init__()

        GN_conv_channels = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 3]
        GN_fc_channels = [2 ** 9]

        # Generator Network
        self.de_trans_1 = tor.nn.ConvTranspose2d(in_channels=GN_fc_channels[0], out_channels=GN_conv_channels[0], kernel_size=32, stride=1, bias=False)
        self.de_conv_1 = self.conv(GN_conv_channels[0], GN_conv_channels[1], 3, 1)
        self.de_bn_1 = tor.nn.BatchNorm2d(GN_conv_channels[1])
        self.de_conv_2 = self.conv(GN_conv_channels[1], GN_conv_channels[2], 3, 1)
        self.de_bn_2 = tor.nn.BatchNorm2d(GN_conv_channels[2])
        self.de_trans_2 = tor.nn.ConvTranspose2d(in_channels=GN_conv_channels[2], out_channels=GN_conv_channels[3], kernel_size=2, stride=2, bias=False)
        #self.out = tor.nn.Sigmoid()
        self.de_conv_3 = self.conv(GN_conv_channels[3], GN_conv_channels[4], 3, 1, relu=False)
        
        self.out = tor.nn.Tanh()


    def forward(self, x) :
        x = x.view(x.size(0), -1, 1, 1)
        x = self.de_trans_1(x)
        x = self.de_conv_1(x)
        x = self.de_bn_1(x)
        x = self.de_conv_2(x)
        x = self.de_bn_2(x)
        x = self.de_trans_2(x)
        x = self.de_conv_3(x)
        out = self.out(x)

        return out


    def load_gn_state(self, state) :
        for i, layer in enumerate(self.state_dict()):
            print("Load dn:", list(state)[i], layer)
            w = state[list(state)[i]]
            self.state_dict()[layer].copy_(w)



class DN(nn.Module) :
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
            #nn.ReLU(inplace=True)
        )
        return fc


    def __init__(self):
        super(DN, self).__init__()
        self.training = True

        DN_conv_channels = [3, 2 ** 6, 2 ** 7]
        DN_fc_channels = [32 * 32 * DN_conv_channels[-1], 2 ** 10, 2 ** 10, 1]

        # Discriminator Network
        self.dn_conv_1 = self.conv(DN_conv_channels[0], DN_conv_channels[1], 3, 1)
        self.dn_conv_2 = self.conv(DN_conv_channels[1], DN_conv_channels[2], 3, 1)
        self.dn_pool_1 = tor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dn_fc_1 = self.fc(DN_fc_channels[0], DN_fc_channels[1])
        self.dn_fc_2 = self.fc(DN_fc_channels[1], DN_fc_channels[2])
        self.d_fc = self.fc(DN_fc_channels[2], DN_fc_channels[3])
        self.c_fc = self.fc(DN_fc_channels[2], DN_fc_channels[3])
        self.drop = tor.nn.Dropout(p=0.5 if self.training else 0)
        self.dn_sig = tor.nn.Sigmoid()


    def forward(self, x) :
        x = self.dn_conv_1(x)
        x = self.dn_conv_2(x)
        x = self.dn_pool_1(x)
        x = x.view(x.size(0), -1)
        x = self.dn_fc_1(x)
        x = self.dn_fc_2(x)
        d = self.d_fc(self.drop(x))
        c = self.c_fc(self.drop(x))
        d = self.dn_sig(d)
        c = self.dn_sig(c)

        return d, c



    def load_dn_state(self, state) :
        for i, layer in enumerate(list(self.state_dict())[:8]) :
            print ("Load dn:", list(state)[i], layer)
            w = state[list(state)[i]]
            self.state_dict()[layer].copy_(w)
