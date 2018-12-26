import torch
import torch.nn as nn

#more dese in fc Layer --> 3 layer
## Model ##
class RelationNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        feature_channel = [3, 128, 256, 512, 512]
        self.conv_feature = nn.Sequential( # Extract Features
            self.conv(feature_channel[0], feature_channel[1], 3),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            self.conv(feature_channel[1], feature_channel[2], 3),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            self.conv( feature_channel[2],feature_channel[3], 3),
            self.conv( feature_channel[3],feature_channel[4], 3)
        )

        score_channel = [feature_channel[-1]*2, 256, 128]
        self.conv_score = nn.Sequential(
            self.conv( score_channel[0], score_channel[1], 3),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            self.conv( score_channel[1], score_channel[2], 3),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        fc_channel = [score_channel[-1]*2*2, 64, 16, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_features= fc_channel[0], out_features= fc_channel[1]),
            nn.ReLU(inplace= True),
            nn.Linear(in_features= fc_channel[1], out_features= fc_channel[2]),
            nn.ReLU(inplace= True),
            nn.Linear(in_features= fc_channel[2], out_features= fc_channel[3]),
            nn.Sigmoid()
        )

    def conv(self,in_channels, out_channels, kernel_size):
        conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels= out_channels,
                     kernel_size= kernel_size, padding= (kernel_size -1)/2),
            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU(inplace= True)
        )
        return conv

    def forward(self, support, query): #for prediction only
        # support shape: (way_num, shot_num, 3,32,32)
        # query shape:   (1, 3,32,32)
        # output shape:  (way_num, 1)
        way_num, shot_num = support.size(0), support.size(1)
        # extract features
        s = support.view(-1,3,32,32)
        s = self.conv_feature(s)
        q = self.conv_feature(query)
        # concatenate
        s = s.view(way_num, shot_num, s.size(1), s.size(2), s.size(3))
        s = torch.mean(s, 1)
        q = q.repeat(way_num,1,1,1)
        x = torch.cat([s,q], 1)
        # compute score
        x = self.conv_score(x)
        x = x.view(way_num, -1)
        score = self.fc(x)
        return score


    def train_one_step(self, support, query):
        # support shape: (way_num, shot_num, 3,32,32)
        # query shape:   (way_num, query_num, 3,32,32)
        # output shape:  (way_num, query_num, way_num, 1)
        way_num, shot_num = support.size(0), support.size(1)
        query_num = query.size(1)
        # extract features
        s = support.view(-1,3,32,32)
        q = query.view(-1,3,32,32)
        s = self.conv_feature(s) # (way*shot, c, )
        q = self.conv_feature(q) # (way*query, c, )

        # concatenate
        s = s.view(way_num, shot_num, s.size(1), s.size(2), s.size(3))
        s = torch.mean(s, 1) # (way, c, )
        s = s.view(1,1, way_num, s.size(1), s.size(2), s.size(3)) 
        s = s.repeat(way_num, query_num, 1,1,1,1)
        q = q.view(way_num, query_num, 1, q.size(1),q.size(2),q.size(3))
        q = q.repeat(1, 1, way_num, 1, 1, 1)
        x = torch.cat([s,q], 3)
        x = x.view(-1,x.size(3),x.size(4),x.size(5))

        # compute score
        x = self.conv_score(x)
        x = x.view(way_num*query_num*way_num, -1)
        score = self.fc(x)
        return score.view(way_num, query_num, way_num, 1)

