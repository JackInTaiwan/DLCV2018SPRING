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

    x_eval_var = Variable(x_eval).type(tor.FloatTensor).cuda()
    y_eval_var = Variable(y_eval).cuda()
    pred = model(x_eval_var)
    pred = tor.max(pred, 1)[1].cuda()
    print (pred[:4])
    print (y_eval[:4])
    correct = int((pred == y_eval_var).data.sum())
    total = int(y_eval_var.size(0) * y_eval_var.size(1) * y_eval_var.size(2))
    acc = round(correct / total, 5)

    return acc
