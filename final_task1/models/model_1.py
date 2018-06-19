import cv2
import torch as tor
import torch.nn as nn
import numpy as np




class MatchNet(nn.Module) :
    def __init__(self):
        super(MatchNet, self).__init__()
        channels = np.array([3, 2 ** 6, 2 ** 6, 2 ** 7, 2 ** 7, 2 ** 8, 2 ** 8, 2 ** 8, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 10])
        channels = [int(num) for num in channels]    # transform type

        self.vgg16 = nn.Sequential(
            self.conv(channels[0], channels[1], 3, 1),
            self.conv(channels[1], channels[2], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(channels[2], channels[3], 3, 1),
            self.conv(channels[3], channels[4], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(channels[4], channels[5], 3, 1),
            self.conv(channels[5], channels[6], 3, 1),
            self.conv(channels[6], channels[7], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(channels[7], channels[8], 3, 1),
            self.conv(channels[8], channels[9], 3, 1),
            self.conv(channels[9], channels[10], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            #self.conv(channels[10], channels[11], 2, 1),
            #self.conv(channels[11], channels[12], 2, 1),
            #self.conv(channels[12], channels[13], 2, 1),
            #nn.MaxPool2d(kernel_size=2),
        )


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


    def flatten(self, x) :
        return x.view(-1)


    def forward(self, x, x_query, y_query) :
        x = x.view(100, 3, 32, 32)
        x = self.vgg16(x)
        x = x.view(20, 5, -1)
        x = tor.mean(x, dim=1)
        print (x.size())
        x_query = self.vgg16(x_query)

        score = tor.mm(x, x_query)
        pred = tor.nn.functional.cosine_similarity(score, x_query)

        return pred
