import numpy as np



def load_data(x_fp, y_fp) :
    x = np.load(x_fp)
    y = np.load(y_fp)
    x_size = x.shape

    return x, y, x_size



def evaluate(model, x_eval, y_eval) :
    import cv2
    import torch as tor
    from torch.autograd import Variable

    correct = 0
    x_eval_var = Variable(x_eval).type(tor.FloatTensor).cuda()
    y_eval_var = Variable(y_eval).cuda()
    for i in range(int(x_eval_var.size(0))) :
        pred = model(x_eval_var[i: i+1])
        pred = tor.max(pred, 1)[1].cuda()
        if 20 < i <25 :
            pred = pred
            print (pred.cpu().data.numpy())
        correct += int((pred == y_eval_var[i:i+1]).data.sum())

    total = int(y_eval_var.size(0) * y_eval_var.size(1) * y_eval_var.size(2))
    acc = round(correct / total, 5)

    return acc
