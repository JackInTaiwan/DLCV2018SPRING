import numpy as np



def load_data(x_fp, y_fp) :
    x = np.load(x_fp)
    y = np.load(y_fp)
    x_size = x.shape

    return x, y, x_size



def evaluate(model, x_var, y_var) :
    import cv2
    import torch as tor

<<<<<<< HEAD
    y_var.cuda()
=======
>>>>>>> fae1bec71839b4addec2002c95a5a9e0d2bc3fac
    pred = model(x_var)
    pred = tor.max(pred, 1)[1].cuda()

    correct = int((pred == y_var).data.sum())
    total = int(y_var.size(0))
    acc = correct / total

    return acc
