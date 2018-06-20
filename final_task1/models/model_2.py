import cv2
import torch as tor
import torch.nn as nn




class RelationNet(nn.Module) :
    def __init__(self, way=20, shot=5):
        super(RelationNet, self).__init__()

        self.way = way
        self.shot = shot

        conv_chls = [3, 2 ** 6, 2 ** 6, 2 ** 7, 2 ** 7, 2 ** 8, 2 ** 8, 2 ** 8, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 9, 2 ** 10]
        vgg16_dense_chls = [conv_chls[10] * 1 * 1, 2 ** 10]

        self.vgg16 = nn.Sequential(
            self.conv(conv_chls[0], conv_chls[1], 3, 1),
            self.conv(conv_chls[1], conv_chls[2], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[2], conv_chls[3], 3, 1),
            self.conv(conv_chls[3], conv_chls[4], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[4], conv_chls[5], 3, 1),
            self.conv(conv_chls[5], conv_chls[6], 3, 1),
            self.conv(conv_chls[6], conv_chls[7], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[7], conv_chls[8], 3, 1),
            self.conv(conv_chls[8], conv_chls[9], 3, 1),
            self.conv(conv_chls[9], conv_chls[10], 3, 1),
            nn.MaxPool2d(kernel_size=4),
            #self.conv(conv_chls[10], conv_chls[11], 2, 1),
            #self.conv(conv_chls[11], conv_chls[12], 2, 1),
            #self.conv(conv_chls[12], conv_chls[13], 2, 1),
            #nn.MaxPool2d(kernel_size=2),
        )

        self.vgg16_dense = self.fc(vgg16_dense_chls[0], vgg16_dense_chls[1], relu=False)

        score_dense_chls = [vgg16_dense_chls[-1] * 2, 2 ** 10, 2 ** 10, 1]

        self.score_dense = nn.Sequential(
            self.fc(score_dense_chls[0], score_dense_chls[1]),
            self.fc(score_dense_chls[1], score_dense_chls[2]),
            self.fc(score_dense_chls[2], score_dense_chls[3], relu=False),
            nn.Sigmoid(),
        )



    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_conv_channels,
                out_channels=out_conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        return conv


    def fc(self, num_in, num_out, relu=True) :
        if relu == True :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=False),
                nn.ReLU(inplace=True)
            )
            return fc
        else :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=False),
            )
            return fc


    def flatten(self, x) :
        return x.view(-1)


    def forward(self, x, x_query) :
        way, shot = x.size(0), x.size(1)
        x = x.view(way * shot, 3, 32, 32)
        x = self.vgg16(x)
        x = x.view(way * shot, -1)
        x = self.vgg16_dense(x)
        x = x.view(self.way, self.shot, -1)
        x = tor.mean(x, dim=1)
        x_query = self.vgg16(x_query)
        x_query = x_query.view(1, -1)
        x_query = self.vgg16_dense(x_query)
        x_query = x_query[0].repeat(x.size(0), 1)

        cat = tor.cat((x, x_query), 1)
        score = self.score_dense(cat)
        return score
