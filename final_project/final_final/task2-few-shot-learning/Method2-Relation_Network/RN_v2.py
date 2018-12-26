import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from matplotlib import pyplot as plt
#from butirecorder import Recorder

###   Parameters  ###
batch_size = 80
lr = 1e-3
save_name = 'rn_0623.pkl'
load_name = 'rn_0623.pkl'

###     Model     ###
# 20 way Few(1,5,10)-shot learning 
class RN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential( ## BEFORE concatenation
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True)
        )

        self.conv2 = nn.Sequential( ##  AFTER concatenation
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size =3,padding= 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc = nn.Sequential( ## Output a score of similarity of two input images
            nn.Linear(in_features=64*2*2,out_features=32),
            nn.ReLU(inplace= True),
            nn.Linear(in_features = 32, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self,x1,x2):
        # Input: 2 image to be compare         shape:(batch size, 3, 32, 32)
        # Output: The rate of similarity (0,1) shape:(batch size, 1)        
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        # concatenate along the depth
        x = torch.cat([x1,x2],1)
        x = self.conv2(x)
        # Flatten
        x = x.view(-1,64*2*2)
        x = self.fc(x)
        return x 
    
###    Functions  ###
def load(name):
    array = np.load(name+'.npy')
    if name == 'test':
        array = np.transpose(array,(0,3,1,2))
    else:
        array = np.transpose(array,(0,1,4,2,3))
    tensor = torch.tensor(array, dtype= torch.float)
    return tensor

def train_one_step(model, optimizer, loss_func, support, target):
    # support & target shape : (class_num, batch_size, 3, 32, 32)
    class_num = support.shape[0]
    pos_label = torch.ones(batch_size,1).cuda()
    neg_label = torch.zeros(batch_size,1).cuda()

    total_loss = 0
    
    for t_id in range(class_num):
        optimizer.zero_grad()
        for s_id in range(class_num):
            in_1 = target[t_id].cuda()
            in_2 = support[s_id].cuda()
            
            out = model(in_1,in_2)
            loss = loss_func( out, pos_label) if t_id == s_id else loss_func( out, neg_label)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()

    return total_loss/(class_num * class_num)
    
def train(retrain ,step_num):
    print('Training...')
    way_num = 5
    rn = RN().cuda()
    rn.train()
    if retrain == True:
        rn.load_state_dict( torch.load(load_name) )
    
    # Materials
    train_data = load('base_train') #(80,500,3,32,32) 
    mse = nn.MSELoss()
    optimizer = optim.SGD(rn.parameters(), lr = lr)

    # Start training
    for step in range(step_num):
        support_ids = torch.randint(500,(batch_size,), dtype = torch.long)
        target_ids = torch.randint(500, (batch_size,), dtype = torch.long)
        shuffle = torch.randperm(80)
        support_set = train_data[shuffle ,support_ids, :,:,:]
        target_set = train_data[shuffle ,target_ids, :,:,:]

        for block_id in range(80//way_num):
            support = support_set[block_id*way_num : (block_id +1)*way_num]
            target = target_set[block_id*way_num : (block_id +1)*way_num]
            loss = train_one_step(rn, optimizer, mse, support,target)
            print('Step ', step+1,'/',step_num,' block',block_id+1,'/',80//way_num,'   loss = ',loss)
        
    torch.save(rn.state_dict(),save_name)

def few_shot_catgory(model, support, target):
    # support:(class num, support size,3,32,32) target: (3,32,32)
    target = target.view(1,3,32,32).cuda()
    class_num, support_size = support.shape[0], support.shape[1]
    score = torch.zeros(1,class_num)

    for class_id in range(class_num):
        for support_id in range(support_size):
            img_s = support[class_id,support_id,:,:,:].view(1,3,32,32).cuda()
            score[0,class_id] += model(img_s, target).cpu().item()
    
    category = torch.max(score,1)[1].item()
    return category


def validation(mode ,support_size): # mode = 'base'/'novel', support_size = shot number
    print('Validation for',support_size,'-shot learning  ...')
    img_valid = load('base_train') if mode == 'base' else load('novel')
    class_num ,img_num = img_valid.shape[0], img_valid.shape[1]
    target_num = 5 #per class
    
    rn = RN().cuda()
    rn.eval()
    rn.load_state_dict(torch.load(load_name))

    target_idx = torch.randint(0,img_num,(target_num,),dtype= torch.long)
    support_idx = torch.arange( support_size ,dtype= torch.long)
    img_target  = img_valid[:,target_idx,:,:,:]
    img_support = img_valid[:,support_idx,:,:,:]
    
    correct = 0
    for class_id in range(class_num):
        print('\nGround Truth:',class_id,' Prediction:',end=' ')
        for target_id in range(target_num):
            predict = few_shot_catgory(rn, img_support, img_target[class_id,target_id])
            print(predict,end=' ')
            if predict == class_id:
                correct += 1
    
    acc = correct/(class_num*target_num)
    print('\n*****************************')
    if mode == 'base':
        print('Accuracy on base class:',acc)
    elif mode == 'novel':
        print('Accuracy on novel class:',acc)
    print('\n*****************************')

def Few_shot(number):
    print(number,'-Shot Learning ...')


def test():
    print('testing...')
    

if __name__ == '__main__':
    print('==========  Relation Network ===========')
    if sys.argv[1] == 'train':
        train(False, int(sys.argv[2]))

    elif sys.argv[1] == 'retrain':
        train(True, int(sys.argv[2]))
    
    elif sys.argv[1] == 'valid_base':
        validation('base',int(sys.argv[2]))
    
    elif sys.argv[1] == 'valid_novel':
        validation('novel',int(sys.argv[2]))

    elif sys.argv[1] == 'few-shot':
        Few_shot(int(sys.argv[2]))
    
    elif sys.argv[1] == 'test':
        test()