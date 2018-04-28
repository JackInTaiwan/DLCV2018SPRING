import numpy as np



def load_data(x_fp, y_fp) :
    x = np.load(x_fp)
    y = np.load(y_fp)
    x_size = x.shape

    return x, y, x_size