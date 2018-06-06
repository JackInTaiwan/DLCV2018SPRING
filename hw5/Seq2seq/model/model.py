import cv2
import torch as tor
import torch.nn as nn




class RNN(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0) :
        super(RNN, self).__init__()

        self.training = True

        self.index = 0
        self.lr = None
        self.lr_decay = None
        self.optim = None
        self.beta = None

        self.epoch = 1
        self.step = 1

        # block_1 LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # block_2 FC Layers
        self.fc_channels = [hidden_size, 2 ** 10, 2 ** 11, 11]

        self.fc_1 = nn.Linear(self.fc_channels[0], self.fc_channels[1])
        self.fc_2 = nn.Linear(self.fc_channels[1], self.fc_channels[2])
        self.fc_3 = nn.Linear(self.fc_channels[2], self.fc_channels[3])
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)



    def forward(self, x, h=None, c=None) :
        o, h = self.lstm(x, (h, c)) if type(h) is tor.Tensor else self.lstm(x)
        o = o[0]
        o = self.relu(o)
        f = self.fc_1(o)
        f = self.drop(self.relu(f)) if self.training else self.relu(f)
        f = self.fc_2(f)
        f = self.drop(self.relu(f)) if self.training else self.relu(f)
        f = self.fc_3(f)
        out = self.sig(f)
        return out, h


    def run_step(self) :
        self.step += 1


    def run_epoch(self) :
        self.epoch += 1


    def train(self) :
        self.training = True


    def eval(self) :
        self.training = False


    def save(self, save_fp) :
        import torch as tor

        tor.save(self, save_fp)

        print ("===== Save Model =====")
        print ("|Model index: {}".format(self.index),
                "\n|Epoch: {}".format(self.epoch),
                "\n|Step: {}".format(self.step),
                "\n|Lr: {} |Lr_decay: {}".format(self.lr, self.lr_decay),
                "\n|Optim: {} |Beta: {}".format(self.optim, self.beta),
                "\n|Save path: {}".format(save_fp),
               )



class RNN_old(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0) :
        super(RNNold, self).__init__()
        self.training = True

        self.index = 0
        self.lr = None
        self.lr_decay = None
        self.optim = None
        self.beta = None

        self.epoch = 1
        self.step = 1

        # block_1 LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # block_2 FC Layers
        self.fc_channels = [hidden_size, 2 ** 10, 2 ** 11, 11]

        self.fc_1 = nn.Linear(self.fc_channels[0], self.fc_channels[1])
        self.fc_2 = nn.Linear(self.fc_channels[1], self.fc_channels[2])
        self.fc_3 = nn.Linear(self.fc_channels[2], self.fc_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()



    def forward(self, x) :
        o, c = self.lstm(x)
        o = o[0][-1]
        o = o.unsqueeze(0)
        f = self.fc_1(o)
        f = self.relu(f)
        f = self.fc_2(f)
        f = self.relu(f)
        f = self.fc_3(f)
        out = self.sig(f)
        return out


    def run_step(self) :
        self.step += 1


    def run_epoch(self) :
        self.epoch += 1


    def save(self, save_fp) :
        import torch as tor

        tor.save(self, save_fp)

        print ("===== Save Model =====")
        print ("|Model index: {}".format(self.index),
                "\n|Epoch: {}".format(self.epoch),
                "\n|Step: {}".format(self.step),
                "\n|Lr: {} |Lr_decay: {}".format(self.lr, self.lr_decay),
                "\n|Optim: {} |Beta: {}".format(self.optim, self.beta),
                "\n|Save path: {}".format(save_fp),
               )
