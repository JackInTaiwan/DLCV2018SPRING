import numpy as np
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier as KNN




"""
model_31 + novel class data features多取不同組合
"""

class Classifier(nn.Module) :
    def __init__(self):
        super(Classifier, self).__init__()

        conv_chls = [3, 2 ** 7, 2 ** 8, 2 ** 8]

        self.vgg16 = nn.Sequential(
            self.conv(conv_chls[0], conv_chls[1], 3, 1),
            nn.BatchNorm2d(num_features=conv_chls[1]),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[1], conv_chls[2], 3, 1),
            nn.BatchNorm2d(num_features=conv_chls[2]),
            nn.MaxPool2d(kernel_size=2),
            self.conv(conv_chls[2], conv_chls[3], 3, 1, relu=False),
            nn.MaxPool2d(kernel_size=2),
            #nn.Tanh(),
        )

        score_dense_chls = [conv_chls[-1] * 4 * 4, 2 ** 10, 80]

        self.fc_1 = self.fc(score_dense_chls[0], score_dense_chls[1], relu=False, sig=True)
        self.fc_2 = self.fc(score_dense_chls[1], score_dense_chls[2], relu=False)
        self.sig = nn.Sigmoid()



    def conv(self, in_conv_channels, out_conv_channels, kernel_size, stride, relu=True):
        if relu :
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_conv_channels,
                    out_channels=out_conv_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
                    bias=True,
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
                    bias=True,
                )
            )
        return conv


    def fc(self, num_in, num_out, sig=False, relu=True) :
        if relu :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=True),
                nn.ReLU(inplace=True)
            )
        elif sig :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=True),
                nn.Sigmoid(),
            )
        else :
            fc = nn.Sequential(
                nn.Linear(num_in, num_out, bias=True),
            )
        return fc


    def flatten(self, x) :
        return x.view(-1)


    def forward(self, x) :
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        score = self.fc_2(x)

        return score


    def pred(self, x_support, x_query) :
        way, shot = int(x_support.size(0)), int(x_support.size(1))

        knn = KNN(
            n_neighbors=1,
        )

        x_support = x_support.view(-1, 3, 32, 32)
        x_support = self.vgg16(x_support)
        x_support = x_support.view(x_support.size(0), -1)
        x_support = self.fc_1(x_support)
        x_support = x_support.view(20, shot, -1)
        x_support = x_support.cpu().detach().numpy()
        x_support_total = [[] for _ in range(20)]
        for i in range(x_support.shape[0]) :
            for j in range(2 ** shot) :
                for k, w in enumerate("{:0>5}".format(bin(j)[2:])) :
                    x_tmp = []
                    if w == "1" :
                        x_tmp.append(x_support[i][k])
                x_tmp = np.array(x_tmp)
                x_tmp = np.mean(x_tmp, axis=1)
                x_support_total[i].append(x_tmp.reshape(-1))

        y_support = np.array([i // x_support_total.shape[1] for i in range(x_support_total.shape[1] * way)])
        x_support = x_support_total.reshape(-1, x_support_total.shape[2])

        knn.fit(x_support, y_support)

        pred_list = []

        for query in x_query.view(-1, 3, 32, 32) :
            query_feature = self.vgg16(query.view(1, 3, 32, 32))
            query_feature = self.fc_1(query_feature.view(query_feature.size(0), -1))
            query_feature = query_feature.cpu().detach().numpy()
            pred = knn.predict(query_feature)
            pred_list.append(int(pred[0]))

        return np.array(pred_list)