import cv2
import torchvision.models
import torch.nn as nn




class RNN(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers=1, lstm_drop=0, fc_drop=0.5) :
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
        self.lstm_drop = lstm_drop

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_drop,
            batch_first=True,
        )

        # block_2 FC Layers
        self.fc_channels = [hidden_size, 2 ** 9, 2 ** 10, 11]

        self.fc_1 = nn.Linear(self.fc_channels[0], self.fc_channels[1])
        self.fc_2 = nn.Linear(self.fc_channels[1], self.fc_channels[2])
        self.fc_3 = nn.Linear(self.fc_channels[2], self.fc_channels[3])
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=fc_drop)
        self.sig = nn.Sigmoid()



    def forward(self, x) :
        o, c = self.lstm(x)
        o = o[0][-1]
        o = o.unsqueeze(0)
        o = self.relu(o)
        f = self.drop(self.fc_1(o)) if self.training == True else self.fc_1(o)
        f = self.relu(f)
        f = self.drop(self.fc_2(f)) if self.training == True else self.fc_2(f)
        f = self.relu(f)
        f = self.drop(self.fc_3(f)) if self.training == True else self.fc_3(f)
        out = self.sig(f)
        return out


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



class Classifier(nn.Module) :
    def __init__(self) :
        super(Classifier, self).__init__()

        self.index = 0
        self.lr = None
        self.lr_decay = None
        self.optim = None
        self.beta = None

        self.epoch = 1
        self.step = 1

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.eval()

        vgg16_fc_channels = [512 * 7 * 10, 2 ** 10]
        self.vgg16 = vgg16.features
        self.vgg16_fc_1 = nn.Linear(vgg16_fc_channels[0], vgg16_fc_channels[1])

        # output = (bs, 512, 7, 10)
        fc_channels = [vgg16_fc_channels[-1], 2 ** 9, 11]

        self.fc_1 = nn.Linear(fc_channels[0], fc_channels[1])
        self.fc_2 = nn.Linear(fc_channels[1], fc_channels[2])

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self, x) :
        x = self.vgg16(x)
        out = x.view(x.size(0), -1)
        out = self.vgg16_fc_1(out)
        return out


    def pred(self, x) :
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sig(x)
        return x


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
