import cv2
import torchvision.models
import torch.nn as nn




class RNN(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0) :
        super(RNN, self).__init__()

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
        )

        # block_2 FC Layers
        self.fc_channels = [input_size, 2 ** 9, 2 ** 10, 11]

        self.fc_1 = nn.Linear(self.fc_channels[0], self.fc_channels[1])
        self.fc_2 = nn.Linear(self.fc_channels[1], self.fc_channels[2])
        self.fc_2 = nn.Linear(self.fc_channels[2], self.fc_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()



    def forward(self, x) :
        x = self.lstm(x)
        f = self.fc_1(x)
        f = self.relu(f)
        f = self.fc_2(f)
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
        print ("|Model index: {} \n \
                |Epoch: {} \n \
                |Step: {} \n \
                |Lr: {} |Lr_decay: {} |\n \
                |Optim: {} |Beta:{} \n \
                |Save path: {}"
               .format(
                    self.index,
                    self.epoch,
                    self.step,
                    self.lr,
                    self.lr_decay,
                    self.optim,
                    self.beta,
                    save_fp,
                )
               )
