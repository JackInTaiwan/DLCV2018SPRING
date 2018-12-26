import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import model_1, model_2

## Parameters ##
WAY = 5
SHOT = 5
LR = 1e-5
CHECK_STEP = 40
SAVE_STEP = 1000


model = model_1
save_name = 'rn0625.pkl'
load_name = 'rn0625.pkl'

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

def sample(data, way, shot):
    # data shape: (class_num, image_num, 3, 32, 32)
    # out = (way, shot, ...), (1,3,32,32), (way, 1)
    way_pick = random.sample(range(data.shape[0]),way)
    shot_pick = random.sample(range(data.shape[1]),shot+1)

    support = data[way_pick][:,shot_pick[:-1]]
    query_pick = random.choice(way_pick)
    query = data[query_pick, shot_pick[-1]].view(1,3,32,32)

    label = torch.zeros((way,1), dtype= torch.float)
    label[way_pick.index(query_pick)] = 1

    return support.cuda(), query.cuda(), label.cuda()


def train(retrain = False, step_num = 5000):
    print('Training...')
    rn = model.RelationNet().cuda()
    rn.train()
    if retrain == True:
        rn.load_state_dict( torch.load(load_name) )

    train_data = load('train')
    mse = nn.MSELoss().cuda()
    optimizer = optim.Adam(rn.parameters(), lr=LR)

    loss_record = 0
    correct = 0

    for step in range(step_num):
        optimizer.zero_grad()
        support, query, label = sample_2(train_data, WAY, SHOT, QUERY)
        
        out = rn(support, query)
        loss = mse( out, label)
        loss.backward()
        loss_record += loss.item()
        optimizer.step()

        if torch.max(out,0)[1] == torch.max(label,0)[1]:
            correct += 1

        if (step+1)%CHECK_STEP == 0:
            print('Step',step+1,'/',step_num,end=' ')
            print('loss = ',loss_record/CHECK_STEP, end=' ')
            print('Acc on base =',correct/CHECK_STEP)
            loss_record = 0; correct = 0

        if (step+1)%SAVE_STEP == 0:
            torch.save(rn.state_dict(),save_name)

def validation(way = WAY, shot= SHOT):
    print('Validation for ',way,'-way',shot,'-shot learning...')
    valid_data = load('valid')
    valid_step = 200
    correct = 0
    rn = model.RelationNet().cuda()
    rn.eval()
    rn.load_state_dict( torch.load(load_name) )

    for step in range(valid_step):
        support, query, label = sample(valid_data, way, shot)
        out = rn(support, query)
        if torch.max(out,0)[1] == torch.max(label,0)[1]:
            correct += 1
    print('Acc on validation set (Base-Test) for',shot,'-shot learning: ',correct/valid_step)
        
def evaluate( shot = SHOT):
    print('Evaluating for ',shot,'-shot learning on novel class...')
    novel_data = load('novel')
    rn = model.RelationNet().cuda()
    rn.eval()
    rn.load_state_dict( torch.load(load_name) )
    
    query_num = 10 #per class
    support_pick = random.sample(range(500),shot)
    support = novel_data[:,support_pick].cuda()
    query_pick = random.sample(range(500),query_num)
    query = novel_data[:,query_pick]
    correct = 0
    
    for class_id in range(20):
        for q_id in range(query_num):
            q = query[class_id, q_id].view(1,3,32,32).cuda()
            score = rn(support, q)
            if torch.max(score,0)[1] == class_id:
                correct += 1

    print('Acc on novel class for',shot,'-shot learning: ',correct/(query_num * 20))

def test():
    print('Testing...')
    evaluate()

##   MAIN   ##
if __name__ == '__main__':
    print('========== Relation Network ver.3 ==========')
    if sys.argv[1] == 'train':
        train(False, int(sys.argv[2]))

    elif sys.argv[1] == 'retrain':
        train(True, int(sys.argv[2]))

    elif sys.argv[1] == 'valid':
        validation(way = 20, shot= 1)
        validation(way = 20, shot= 5)
        validation(way = 20, shot= 10)

    elif sys.argv[1] == 'eval':
        evaluate(shot = 1)
        evaluate(shot = 5)
        evaluate(shot = 10)
    
    elif sys.argv[1] == 'test':
        test()