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
                in_conv_channels=in_conv_channels,
                out_conv_channels=out_conv_channels,
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


    def flatten(self, x):
        return x.view(x.size(0), -1)  # x size = (BS, num_FM, h, w)


    def __init__(self):
        super(AVE, self).__init__()
        self.index = 0

        conv_channels = np.array([3, 2 ** 6, 2 ** 6, 2 ** 7, 2 ** 7, 2 ** 8, 2 ** 8, 2 ** 8, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 10])
        conv_channels = [int(num) for num in conv_channels]    # transform type
        fc_channels = np.array([2 ** 7, 2 ** 8, 2 ** 9, 10])
        fc_channels = [int(num) for num in fc_channels]  # transform type

        # block 1
        self.b_1_conv_1 = self.conv(conv_channels[0], conv_channels[1], 3, 1)
        self.b_1_conv_2 = self.conv(conv_channels[1], conv_channels[2], 3, 1)
        self.b_1_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 2
        self.b_2_conv_1 = self.conv(conv_channels[2], conv_channels[3], 3, 1)
        self.b_2_conv_2 = self.conv(conv_channels[3], conv_channels[4], 3, 1)
        self.b_2_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 3
        self.b_3_fc_1 = self.fc(fc_channels[0], fc_channels[1])
        self.b_3_fc_2 = self.fc(fc_channels[1], fc_channels[2])
        self.b_3_fc_3 = self.fc(fc_channels[2], fc_channels[3])

    def forward(self, x):
        x = self.b_1_conv_1(x)
        x = self.b_1_conv_2(x)
        x = self.b_1_pool_1(x)

        x = self.b_2_conv_1(x)
        x = self.b_2_conv_2(x)
        x = self.b_2_pool_1(x)

        x = self.flatten(x)
        x = self.b_3_fc_1(x)
        x = self.b_3_fc_2(x)
        x = self.b_3_fc_2(x)

        return x


    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.001)
            tor.nn.init.normal(m.bias, 0, 0.001)
        elif classname.find("Conv") != -1 and self.index >=44 :
            m.weight.data.normal_(0.01, 0.001)
            m.bias.data.normal_(0.01, 0.001)
        self.index += 1



class GN(nn.Module):
    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_conv_channels=in_conv_channels,
                out_conv_channels=out_conv_channels,
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

        conv_channels = np.array([3, 2 ** 6, 2 ** 6, 2 ** 7, 2 ** 7, 2 ** 8, 2 ** 8, 2 ** 8, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 10])
        conv_channels = [int(num) for num in conv_channels]    # transform type
        fc_channels = np.array([2 ** 7, 2 ** 8, 2 ** 9, 10])
        fc_channels = [int(num) for num in fc_channels]  # transform type

        # block 1
        self.b_1_conv_1 = self.conv(conv_channels[0], conv_channels[1], 3, 1)
        self.b_1_conv_2 = self.conv(conv_channels[1], conv_channels[2], 3, 1)
        self.b_1_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 2
        self.b_2_conv_1 = self.conv(conv_channels[2], conv_channels[3], 3, 1)
        self.b_2_conv_2 = self.conv(conv_channels[3], conv_channels[4], 3, 1)
        self.b_2_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 3
        self.b_3_fc_1 = self.fc(fc_channels[0], fc_channels[1])
        self.b_3_fc_2 = self.fc(fc_channels[1], fc_channels[2])
        self.b_3_fc_3 = self.fc(fc_channels[2], fc_channels[3])

    def forward(self, x):
        x = self.b_1_conv_1(x)
        x = self.b_1_conv_2(x)
        x = self.b_1_pool_1(x)

        x = self.b_2_conv_1(x)
        x = self.b_2_conv_2(x)
        x = self.b_2_pool_1(x)

        x = self.flatten(x)
        x = self.b_3_fc_1(x)
        x = self.b_3_fc_2(x)
        x = self.b_3_fc_2(x)

        return x


    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.001)
            tor.nn.init.normal(m.bias, 0, 0.001)
        elif classname.find("Conv") != -1 and self.index >=44 :
            m.weight.data.normal_(0.01, 0.001)
            m.bias.data.normal_(0.01, 0.001)
        self.index += 1
