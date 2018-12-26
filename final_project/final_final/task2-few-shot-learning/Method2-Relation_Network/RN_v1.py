import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from matplotlib import pyplot as plt
#from butirecorder import Recorder

###   Parameters  ###
epoch_num = 20
batch_size = 80
data_len = 10000
check_step = 25
lr = 1e-4
save_name = 'rn_0623.pkl'
load_name = 'rn_0623.pkl'
torch.manual_seed(851004)

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

def get_train_data():
    pair_num = data_len//80 # number of images sampled form each class
    base_train = load('base_train') # shape = (80,500,3,32,32)
    rand_id_1 = torch.randint(0,500,(pair_num,), dtype = torch.long)
    rand_id_2 = torch.randint(0,500,(pair_num,), dtype = torch.long)
    perm_id = torch.randperm(data_len)
    
    # get two "same class" array
    same_1 = torch.zeros((data_len ,3,32,32), dtype = torch.float)
    same_2 = torch.zeros((data_len ,3,32,32), dtype = torch.float)
    for img_class in range(80):
        same_1[ img_class*pair_num : (img_class +1)*pair_num ,:] = base_train[ img_class, rand_id_1,:]
        same_2[ img_class*pair_num : (img_class +1)*pair_num ,:] = base_train[ img_class, rand_id_2,:]
    same_1 = same_1[perm_id]
    same_2 = same_2[perm_id]

    # get two "different class" array
    diff_1 = same_1
    diff_2 = same_2[torch.randperm(data_len)]
    
    return same_1, same_2, diff_1, diff_2
    
def train(retrain = False):
    print('Training...')
    rn = RN().cuda()
    rn.train()
    if retrain == True:
        rn.load_state_dict( torch.load(load_name) )
    
    # Loss, Optimizer, Ground Truth Labels 
    mse = nn.MSELoss()
    optimizer = optim.Adam(rn.parameters(), lr = lr)
    pos_label = torch.ones((batch_size,1), dtype= torch.float).cuda()
    neg_label = torch.zeros((batch_size,1), dtype= torch.float).cuda()

    # Start training
    for epoch in range(epoch_num):
        same_1, same_2, diff_1, diff_2 = get_train_data()
        same_class_loss = 0
        diff_class_loss = 0

        for batch_id in range(data_len//batch_size):
            optimizer.zero_grad()
            # same class 
            in_1 = same_1[batch_id * batch_size : (batch_id +1)*batch_size].cuda()
            in_2 = same_2[batch_id * batch_size : (batch_id +1)*batch_size].cuda()
            out = rn(in_1,in_2)
            loss_same = mse(out,pos_label)
            loss_same.backward() 
            same_class_loss += loss_same.item()
            out = rn(in_2,in_1)
            loss_same = mse(out,pos_label)
            loss_same.backward() 
            same_class_loss += loss_same.item()
            # different class
            in_1 = diff_1[batch_id * batch_size : (batch_id +1)*batch_size].cuda()
            in_2 = diff_2[batch_id * batch_size : (batch_id +1)*batch_size].cuda()
            out = rn(in_1,in_2)
            loss_diff = mse(out,neg_label)
            loss_diff.backward()
            diff_class_loss += loss_diff.item()
            out = rn(in_2,in_1)
            loss_diff = mse(out,neg_label)
            loss_diff.backward()
            diff_class_loss += loss_diff.item()
            
            optimizer.step()

            if (batch_id+1)%check_step == 0:
                print("Epoch",epoch+1,'/',epoch_num," finished ",batch_id+1,"/",data_len/batch_size)
                print("Same-class Loss:      ",same_class_loss/check_step)
                print("Different-class Loss: ",diff_class_loss/check_step)
                print("Average Loss:         ",(same_class_loss+diff_class_loss)/(2*check_step),'\n')
                same_class_loss = 0
                diff_class_loss = 0
        
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
    a, b, c, d = get_train_data()
    x = np.transpose(b[456].numpy(),(1,2,0))
    plt.imshow(x)
    plt.show()
    y = np.transpose(a[456].numpy(),(1,2,0))
    plt.imshow(y)
    plt.show()

if __name__ == '__main__':
    print('==========  Relation Network ===========')
    if sys.argv[1] == 'train':
        train(retrain= False)

    elif sys.argv[1] == 'retrain':
        train(retrain= True)
    
    elif sys.argv[1] == 'valid_base':
        validation('base',int(sys.argv[2]))
    
    elif sys.argv[1] == 'valid_novel':
        validation('novel',int(sys.argv[2]))

    elif sys.argv[1] == 'few-shot':
        Few_shot(int(sys.argv[2]))
    
    elif sys.argv[1] == 'test':
        test()