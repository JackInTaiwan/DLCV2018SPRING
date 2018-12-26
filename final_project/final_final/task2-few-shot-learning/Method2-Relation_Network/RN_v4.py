import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import model_1, model_2,model_3,model_4, model_5, model_6, model_7,model_8,model_9,model_11, model_12, model_13,model_14
from matplotlib import pyplot as plt
import imutils as imu
random.seed(10)
## Parameters ##
WAY = 5
SHOT = 10
QUERY = 10
LR = 1e-4
CHECK_STEP = 40
SAVE_STEP = 500


model = model_14
model_ver = 'm14'
base_save = 'rn_base.pkl'; base_load = 'rn_base.pkl'
one_save = 'rn_1_0630_'+model_ver+'.pkl'; one_load = 'rn_1_0630_'+model_ver+'.pkl'
five_save = 'rn_5_0630_'+model_ver+'.pkl'; five_load = 'rn_5_0630_'+model_ver+'.pkl'
ten_save = 'rn_10_0630_'+model_ver+'.pkl'; ten_load = 'rn_10_0630_'+model_ver+'.pkl'

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

def get_rotate(imgs, angle):
    # imgs' shape(class_num, img_num, 32,32,3)numpy , output is in the same shape
    class_num, img_num = imgs.shape[0], imgs.shape[1]
    rotate = np.zeros((class_num, img_num, 32,32,3))
    for c_id in range(class_num):
        for i_id in range(img_num):
            rotate[c_id,i_id] = imu.rotate(imgs[c_id,i_id],angle)
    return rotate

def get_flip(imgs,mode = 'h'):
    # imgs' shape(class_num, img_num, 32,32,3)numpy , output is in the same shape
    # mode = 'v' -> flip vertically, mode = 'h' -> flip horizontally
    class_num, img_num = imgs.shape[0], imgs.shape[1]
    axis = 0 if mode == 'v' else 1 #flip axis
    flip = np.zeros((class_num, img_num, 32,32,3))
    for c_id in range(class_num):
        for i_id in range(img_num):
            flip[c_id,i_id] = np.flip(imgs[c_id,i_id], axis)
    return flip

def load_few_novel(shot):
    # return a few-shot image for training in shape of (20,500,3,32,32) -> torch
    novel = np.load('novel.npy') #(20,500,32,32,3)
    few_o = novel[:,:shot, :,:,:]
    few_f = get_flip(few_o)
    few = torch.cat([torch.tensor(few_o,dtype=torch.float),torch.tensor(few_f,dtype=torch.float)], 1)
    '''
    angles = [5,-5,10,-10,]
    for angle in angles:
        f_or = torch.tensor(get_rotate(few_o,angle),dtype=torch.float)
        f_fr = torch.tensor(get_rotate(few_f,angle),dtype=torch.float)
        few = torch.cat([few,f_or,f_fr], 1)
    '''
    few = few.permute(0,1,4,2,3) 
    few = few.repeat(1,500//few.shape[1], 1,1,1)
    if few.shape[1] != 500:
        few_remain = few[:,:(500-few.shape[1]),:,:,:]
        few = torch.cat([few,few_remain],1)
    return few

def load_train_data(mode):
    #return shape:(80,500, ...) or (100,500, ...)
    base = load('train')
    novel = 0
    if mode == 'base':
        return base
    elif mode == 'one':
        novel = load_few_novel(1)
    elif mode == 'five':
        novel = load_few_novel(5)
    elif mode == 'ten':
        novel = load_few_novel(10)
    data = torch.cat([base,novel], 0)
    return data
    

def sample_train(data, way, shot, query):
    # data shape: (class_num, image_num, 3, 32, 32)
    # out = (way, shot, ...), (way, query, ...), (way, query, way, 1)
    way_pick = random.sample(range(data.shape[0]),way)
    shot_pick = random.sample(range(data.shape[1]),shot)
    query_pick = random.sample(range(data.shape[1]), query)

    support_set = data[way_pick][:,shot_pick]
    query_set   = data[way_pick][:,query_pick]

    #return support_set, query_set, label
    return support_set, query_set

def sample_predict(data, way, shot):
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


def train(mode , retrain = False, step_num = 5000):
    print('Training',mode,'...')
    rn = model.RelationNet()
    rn.train()

    #if mode != 'base' and retrain == False:
    #    load_model('base',rn)
    if retrain == True:
        load_model(mode,rn)
    rn = rn.cuda()

    train_data = load_train_data(mode).cuda()
    mse = nn.MSELoss().cuda()
    optimizer = optim.Adam(rn.parameters(), lr=LR)

    loss_record = 0
    correct = 0
    
    label = torch.zeros((WAY,QUERY, WAY, 1), dtype= torch.float).cuda()
    for i in range(WAY):
        label[i,:,i,0] = 1
    gt = torch.max(label,2)[1].cuda()
    novel = load('novel')

    for step in range(step_num):
        optimizer.zero_grad()
        support, query = sample_train(train_data, WAY, SHOT, QUERY)
        
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
            rn.eval()
            evaluate(mode,rn,novel)
            save_model(mode,rn)
            rn.train()
            
def validation(mode, rn, valid_data, way = WAY, shot= SHOT):
    print('Validation for ',way,'-way',shot,'-shot learning...')
    valid_step = 200
    correct = 0

    for step in range(valid_step):
        support, query, label = sample_predict(valid_data, way, shot)
        out = rn(support, query)
        if torch.max(out,0)[1] == torch.max(label,0)[1]:
            correct += 1
    print('Acc on validation set (Base-Test) for',shot,'-shot learning: ',correct/valid_step)
        
def evaluate(mode,rn,novel_data, shot = 0):
    print('Evaluating for ',mode,'-shot learning on novel class...')
    
    if mode == 'one': shot = 1
    elif mode == 'five': shot = 5
    elif mode == 'ten':shot = 10
    
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
    print()

    
def generate(mode, model, novel, test):
    shot_num = 0
    if mode == 'one': shot_num = 1
    elif mode == 'five': shot_num = 5
    elif mode == 'ten': shot_num = 10

    name = 'submission_'+str(shot_num)+'_shot.csv'
    file = open(name,'w')
    file.write('image_id,predicted_label\n')
    support_pick = random.sample(range(500), shot_num)
    support = novel[:, support_pick ].cuda()
    
    test_num = test.shape[0]
    predict = torch.zeros((test_num,1), dtype= torch.int)
    for t_id in range(test_num):
        img = test[t_id].view(1,3,32,32).cuda()
        score = model(support, img)
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


def generate_csv(mode):
    print('Generating result...')
    novel = load('novel')
    test  = load('test')
    rn = model.RelationNet().cuda()
    rn.eval()
    load_model(mode,rn)
    generate(mode,rn,novel,test)
    

def test():
    print('Testing...')
    few = load_few_novel(5)
    print(few.shape)


##   MAIN   ##
if __name__ == '__main__':
    print('========== Relation Network ver.4 ==========')
    if sys.argv[1] == 'train':
        mode = sys.argv[2]; step = int(sys.argv[3])
        train(mode, False, step)

    elif sys.argv[1] == 'retrain':
        mode = sys.argv[2]; step = int(sys.argv[3])
        train(mode,True, step)

    elif sys.argv[1] == 'valid':
        mode = sys.argv[2]
        valid_data = load('valid')
        rn = model.RelationNet().cuda()
        rn.eval()
        load_model(mode,rn)
        
        validation(mode,rn,valid_data,way = 20, shot= 1)
        validation(mode,rn,valid_data,way = 20, shot= 5)
        validation(mode,rn,valid_data,way = 20, shot= 10)
        
    elif sys.argv[1] == 'eval':
        mode = sys.argv[2]
        novel_data = load('novel')
        rn = model.RelationNet().cuda()
        rn.eval()
        load_model(mode,rn)
        if mode == 'base':
            evaluate('base',rn,novel_data,shot = 1)
            evaluate('base',rn,novel_data,shot = 5)
            evaluate('base',rn,novel_data,shot = 10)
        else:
            evaluate(mode,rn,novel_data)

    elif sys.argv[1] == 'generate':
        mode = sys.argv[2]
        generate_csv(mode)
    
    elif sys.argv[1] == 'test':
        test()