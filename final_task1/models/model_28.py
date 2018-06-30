import torch as tor
import torch.nn as nn




class Classifier(nn.Module) :
    def __init__(self):
        super(Classifier, self).__init__()

        conv_chls = [3, 2 ** 6, 2 ** 7, 2 ** 8]

        self.vgg16 = nn.Sequential(
            self.conv(conv_chls[0], conv_chls[1], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[1], conv_chls[2], 3, 1),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[2], conv_chls[3], 3, 1, relu=False),
            nn.MaxPool2d(kernel_size=4),
        )

        score_dense_chls = [conv_chls[-1] * 2 * 2, 2 ** 10, 100]

        self.score_dense = nn.Sequential(
            self.fc(score_dense_chls[0], score_dense_chls[1]),
            self.fc(score_dense_chls[1], score_dense_chls[2], relu=False),
            nn.Sigmoid(),
        )



    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride, relu=True):
        if relu :
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
        else :
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_conv_channels,
                    out_channels=out_conv_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
                    bias=False,
                )
            )
        return conv


    def fc(self, num_in, num_out, sig=False, relu=True) :
        if relu :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=False),
                nn.ReLU(inplace=True)
            )
        elif sig :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=False),
                nn.Sigmoid(),
            )
        else :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=False),
            )
        return fc


    def flatten(self, x) :
        return x.view(-1)


    def forward(self, x) :
            x = self.vgg16(x)
            x = x.view(x.size(0), -1)
            score = self.score_dense(x)
            return score
