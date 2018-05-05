import numpy as np



def load_data(x_fp) :
    x = np.load(x_fp)
    x_size = x.shape

    return x, x_size




def save_pic(save_fp, model, pic_n) :
    import cv2
    import os
    import matplotlib.pyplot as plt
    import torch as tor
    from torch.autograd import Variable

    for i in range(pic_n) :
        file = "{0:>5}.png".format(i)
        file_fp = os.path.join("./hw4_data", file)

        img = np.array([plt.imread(file_fp)]) / 255.
        img_var = Variable(tor.FloatTensor(img))
        img_var = img_var.permute(0, 3, 1, 2)
        out = model(img_var)
        out = out.permute(0, 2, 3, 1)
        out_img = out.data.numpy()

        plt.imsave(os.path.join(save_fp, file), out_img)

        print ("|Output {} is saved.".format(file))




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
        correct += int((pred == tor.max(y_eval_var[i:i+1],1)[1]).data.sum())

    total = int(y_eval_var.size(0) * y_eval_var.size(1) * y_eval_var.size(2))
    acc = round(correct / total, 5)

    return acc
