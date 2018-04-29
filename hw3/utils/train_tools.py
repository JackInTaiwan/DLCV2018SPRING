import numpy as np



def load_data(x_fp, y_fp) :
    x = np.load(x_fp)
    y = np.load(y_fp)
    x_size = x.shape

    return x, y, x_size



def evaluate(model, x_var, y_var) :
    import cv2
    import torch as tor

    pred = model(x_var)
    pred = tor.max(pred, 1)[1].cuda()

    correct = int((pred == y_var).data.sum())
    total = int(y_var.size(0) * y_var.size(1) * y_var.size(2))
    acc = correct / total

    return acc
