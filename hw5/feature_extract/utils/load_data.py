import numpy as np




def normalize(data) :
    """
    :param data: shape = (n, m, h, w, 3)
                 where values is required to be (0, 1)
    :return: new data
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for n in range(data.shape[0]) :
        for i in range(3) :
            data[n][:, :, :, i] = (data[n][:, :, :, i] - mean[i]) / std[i]

    return data