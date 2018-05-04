import cv2
import torch as tor
import torch.nn as nn
import numpy as np



""" Parameters """
VGG16_PRETRAINED_FP = "./models/vgg16_pretrained.h5"



""" Module Build """
class FCN(nn.Module):
    def conv(self, in_channels, out_channels, kernel_size, stride, bias=True):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
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
        super(FCN, self).__init__()
        self.index = 0

        channels = np.array([3, 2 ** 6, 2 ** 6, 2 ** 7, 2 ** 7, 2 ** 8, 2 ** 8, 2 ** 8, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 10])
        channels = [int(num) for num in channels]    # transform type

        # block 1
        self.b_1_conv_1 = self.conv(channels[0], channels[1], 3, 1)
        self.b_1_conv_2 = self.conv(channels[1], channels[2], 3, 1)
        self.b_1_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 2
        self.b_2_conv_1 = self.conv(channels[2], channels[3], 3, 1)
        self.b_2_conv_2 = self.conv(channels[3], channels[4], 3, 1)
        self.b_2_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 3
        self.b_3_conv_1 = self.conv(channels[4], channels[5], 3, 1)
        self.b_3_conv_2 = self.conv(channels[5], channels[6], 3, 1)
        self.b_3_conv_3 = self.conv(channels[6], channels[7], 3, 1)
        self.b_3_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 4
        self.b_4_conv_1 = self.conv(channels[7], channels[8], 3, 1)
        self.b_4_conv_2 = self.conv(channels[8], channels[9], 3, 1)
        self.b_4_conv_3 = self.conv(channels[9], channels[10], 3, 1)
        self.b_4_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 5
        self.b_5_conv_1 = self.conv(channels[10], channels[11], 3, 1)
        self.b_5_conv_2 = self.conv(channels[11], channels[12], 3, 1)
        self.b_5_conv_3 = self.conv(channels[12], channels[13], 3, 1)
        self.b_5_pool_1 = nn.MaxPool2d(kernel_size=2)
        # block 6
        self.b_6_conv_1 = self.conv(channels[13], channels[14], 7, 1)
        self.b_6_conv_2 = self.conv(channels[14], channels[14], 1, 1)
        self.b_6_conv_3 = self.conv(channels[14], 7, 1, 1)
        # block 7
        self.b_7_trans_1 = nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=137, stride=25, bias=False) # f.m. size = (16, 16)
        # block 8
        #self.b_8_softmax_1 = nn.Softmax(dim=1)
        self.sigmoid = tor.nn.Sigmoid()

    def forward(self, x):
        b_1_conv_1 = self.b_1_conv_1(x)
        b_1_conv_2 = self.b_1_conv_2(b_1_conv_1)
        b_1_pool_1 = self.b_1_pool_1(b_1_conv_2)
        b_2_conv_1 = self.b_2_conv_1(b_1_pool_1)
        b_2_conv_2 = self.b_2_conv_2(b_2_conv_1)
        b_2_pool_1 = self.b_2_pool_1(b_2_conv_2)
        b_3_conv_1 = self.b_3_conv_1(b_2_pool_1)
        b_3_conv_2 = self.b_3_conv_2(b_3_conv_1)
        b_3_conv_3 = self.b_3_conv_3(b_3_conv_2)
        b_3_pool_1 = self.b_3_pool_1(b_3_conv_3)
        b_4_conv_1 = self.b_4_conv_1(b_3_pool_1)
        b_4_conv_2 = self.b_4_conv_2(b_4_conv_1)
        b_4_conv_3 = self.b_4_conv_3(b_4_conv_2)
        b_4_pool_1 = self.b_4_pool_1(b_4_conv_3)
        b_5_conv_1 = self.b_5_conv_1(b_4_pool_1)
        b_5_conv_2 = self.b_5_conv_2(b_5_conv_1)
        b_5_conv_3 = self.b_5_conv_3(b_5_conv_2)
        b_5_pool_1 = self.b_5_pool_1(b_5_conv_3)
        b_6_conv_1 = self.b_6_conv_1(b_5_pool_1)
        b_6_conv_2 = self.b_6_conv_2(b_6_conv_1)
        b_6_conv_3 = self.b_6_conv_3(b_6_conv_2)
        b_7_tran_1 = self.b_7_trans_1(b_6_conv_3)
        #b_8_softmax_1 = self.b_8_softmax_1(b_7_tran_1)
        out = self.sigmoid(b_7_tran_1)

        #return b_8_softmax_1
        #return b_7_tran_1
        return out


    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.001)
            tor.nn.init.normal(m.bias, 0, 0.001)
        elif classname.find("Conv") != -1 and self.index >=44 :
            m.weight.data.normal_(0.00, 0.001)
            m.bias.data.normal_(0.00, 0.001)
        self.index += 1


    def params_permute(self, m) :
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 and self.index < 44:
            m.weight.data.copy_(m.weight.data.permute(1, 0, 3, 2))

        self.index += 1


    def all_init(self) :
        self.apply(self.params_init)


    def all_permute(self) :
        self.apply(self.params_permute)


    def vgg16_init(self) :
        import h5py
        layers = list(self.state_dict().keys())
        index = 0

        data = h5py.File(VGG16_PRETRAINED_FP)
        for layer in list(data.keys())[:-4] :

            for ele in data[layer].keys() :
                weights = np.array(data[layer][ele])
                weights = tor.FloatTensor(weights)
                if "_b_" not in ele :
                    weights = weights.permute(3, 2, 1, 0)
                print (layers[index])
                self.state_dict()[layers[index]].copy_(weights)
                index += 1


    def vgg16_load(self, state_dict) :
        for item in list(state_dict)[:-1] :
            self.state_dict()[item].copy_(state_dict[item])