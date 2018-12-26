import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import model_1, model_2,model_3,model_4, model_5, model_6, model_7,model_8,model_9,model_11, model_12, model_13,model_14
from matplotlib import pyplot as plt
random.seed(10)
## Parameters ##
## for trianing ##
WAY = 5
SHOT = 10
QUERY = 10
LR = 1e-4
CHECK_STEP = 50
SAVE_STEP = 500


model = model_13
model_ver = 'm13'
base_save = 'rn_base.pkl'; base_load = 'rn_base.pkl'
one_save = 'rn_1_v6_'+model_ver+'.pkl'; one_load = 'rn_1_v6_'+model_ver+'.pkl'
five_save = 'rn_5_v6_'+model_ver+'.pkl'; five_load = 'rn_5_v6_'+model_ver+'.pkl'
ten_save = 'rn_10_v6_'+model_ver+'.pkl'; ten_load = 'rn_10_v6_'+model_ver+'.pkl'

## Functions ##

def load(mode):
    data = 0
    if mode == 'train':
        data = torch.tensor(np.load('base_train.npy'), dtype=torch.float).permute(0,1,4,2,3) 
    elif mode == 'valid':
        data = torch.tensor(np.load('base_valid.npy'), dtype=torch.float).permute(0,1,4,2,3)
    elif mode == 'novel':
        data = torch.tensor(np.load('novel.npy'), dtype=torch.float).permute(0,1,4,2,3)
    elif mode == 'test':
        data = torch.tensor(np.load('test.npy'), dtype=torch.float).permute(0,3,1,2)
    return data

def flip(tensor):
    # input:(way, img_num, ...) output:(way, img_num*3, ...)
    flip = torch.arange(31,-1,-1).long()
    tensor_h = tensor[:,:,:,:,flip]
    tensor_v = tensor[:,:,:,flip,:]
    tensor = torch.cat([tensor, tensor_v, tensor_h], 1)
    return tensor

def load_train_data(support_set):
    train = load('train')
    novel = support_set.repeat(1,500//support_set.shape[1], 1,1,1)
    train_data = torch.cat([train,novel], 0)
    return flip(train_data)

def sample_train(data):
    # data shape: (class_num, image_num, 3, 32, 32)
    # out = (way, shot, ...), (way, query, ...)
    way_pick = random.sample(range(data.shape[0]),WAY)
    shot_pick = random.sample(range(data.shape[1]),SHOT)
    query_pick = random.sample(range(data.shape[1]),QUERY)

    support = data[way_pick][:,shot_pick]
    query   = data[way_pick][:,query_pick]
    
    return support.cuda(), query.cuda()


def save_model(mode, rn):
    if mode == 'base':
        torch.save(rn.state_dict(),base_save)
    elif mode == 'one':
        torch.save(rn.state_dict(),one_save)
    elif mode == 'five':
        torch.save(rn.state_dict(),five_save)
    elif mode == 'ten':
        torch.save(rn.state_dict(),ten_save)

def load_model(mode, rn):
    if mode == 'base':
        rn.load_state_dict( torch.load(base_load) )
    elif mode == 'one':
        rn.load_state_dict( torch.load(one_load) )
    elif mode == 'five':
        rn.load_state_dict( torch.load(five_load) )
    elif mode == 'ten':
        rn.load_state_dict( torch.load(ten_load) )


def train(rn, support_set, step_num = 5000):
    print('Training',mode,'...')
    train_data = load_train_data(support_set)
    
    mse = nn.MSELoss().cuda()
    optimizer = optim.Adam(rn.parameters(), lr=LR)

    label = torch.zeros((WAY,QUERY, WAY, 1), dtype= torch.float).cuda()
    for i in range(WAY):
        label[i,:,i,0] = 1
    gt = torch.max(label,2)[1].cuda()
    
    loss_record = 0
    correct = 0
    
    for step in range(step_num):
        optimizer.zero_grad()
        support, query = sample_train(train_data)
        
        score = rn.train_one_step(support, query)
        loss = mse( score, label)
        loss.backward()
        loss_record += loss.item()
        optimizer.step()

        score = torch.max(score,2)[1]
        
        correct += torch.tensor(score[ score == gt].size()).item()

        if (step+1)%CHECK_STEP == 0:
            print('Step',step+1,'/',step_num,end=' ')
            print('loss = ',loss_record/CHECK_STEP, end=' ')
            print('Acc on train =',correct/(CHECK_STEP*WAY*QUERY))
            loss_record = 0; correct = 0

        if (step+1)%SAVE_STEP == 0:
            save_model(mode,rn)
            rn.eval()
            evaluate(rn, support_set)
            rn.train()
            
        
def evaluate( rn,support_set):

    support = flip(support_set).cuda()
    gt = torch.zeros(20,QUERY, 1).long()
    for i in range(20):
        gt[i, :,0] = i

    novel_data = load('novel')
    correct = 0

    for i in range(500//QUERY):
        q = novel_data[:,(i*QUERY):(i+1)*QUERY].cuda()
        score = rn.train_one_step(support, q)
        score = torch.max(score,2)[1].cpu()
        correct += torch.tensor(score[ score == gt].size()).item()

    print('Acc on novel class : ',correct/(500 * 20),'\n')
    
    
def generate(mode, rn, support_set):
    
    name = 'submission_'+mode+'_shot.csv'
    file = open(name,'w')
    file.write('image_id,predicted_label\n')
    
    support_set = support_set.cuda()
    test = load('test')
    test_num = test.shape[0]
    predict = torch.zeros((test_num,1), dtype= torch.int)

    for t_id in range(test_num):
        img = test[t_id].view(1,3,32,32).cuda()
        score = rn(support_set, img)
        predict[ t_id ] = torch.max(score,0)[1]

    predict[ predict == 1] = 10
    predict[ predict == 2] = 23 
    predict[ predict == 3] = 30
    predict[ predict == 4] = 32 
    predict[ predict == 5] = 35 
    predict[ predict == 6] = 48 
    predict[ predict == 7] = 54 
    predict[ predict == 8] = 57 
    predict[ predict == 9] = 59 
    predict[ predict == 10] = 60 
    predict[ predict == 11] = 64 
    predict[ predict == 12] = 66 
    predict[ predict == 13] = 69 
    predict[ predict == 14] = 71 
    predict[ predict == 15] = 82 
    predict[ predict == 16] = 91 
    predict[ predict == 17] = 92 
    predict[ predict == 18] = 93 
    predict[ predict == 19] = 95 
    
    for i in range(test_num):
        if predict[i].item() == 0:
            file.write(str(i)+',00\n')
        else:
            file.write(str(i)+','+str(predict[i].item())+'\n')

    file.close()
    print('Prediction result:',name,'is generated!')



##   MAIN   ##
if __name__ == '__main__':
    print('========== Relation Network ver.5 ==========')
    
    mode = sys.argv[2]
    shot = 0
    if mode == 'one': shot = 1
    elif mode == 'five': shot = 5
    elif mode == 'ten': shot = 10
    support_pick = random.sample(range(500),shot)
    support_set = load('novel')[:,support_pick]
    

    if sys.argv[1] == 'train':
        step = int(sys.argv[3])
        rn = model.RelationNet()
        rn.train()
        rn = rn.cuda()
        train(rn,support_set, step)

    elif sys.argv[1] == 'retrain':
        step = int(sys.argv[3])
        rn = model.RelationNet()
        load_model(mode, rn)
        rn.train()
        rn = rn.cuda()
        train(rn,support_set, step)
        
    elif sys.argv[1] == 'eval':
        rn = model.RelationNet()
        load_model(mode,rn)
        rn.eval()
        rn = rn.cuda()
        evaluate(rn,support_set)

    elif sys.argv[1] == 'generate':
        rn = model.RelationNet()
        load_model(mode,rn)
        rn.eval()
        rn = rn.cuda()
        generate(mode,rn,support_set)
    
    elif sys.argv[1] == 'test':
        print('test')
        t = load('novel')
        t = flip(t)
        print(t.shape) 