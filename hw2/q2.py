import scipy.io
import argparse
import cv2
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.signal import convolve2d



def color_segmentation(img, color_mode="rgb") :
    if color_mode == "lab" :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # shape = (h, w, cha)
    h, w, cha = img.shape
    data = img.reshape(h * w, cha)

    k, max_iter = 10, 1000
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans.fit(data)
    y = kmeans.predict(data)
    y = y.reshape(h, w)
    print (y)
    plt.imshow(y)
    plt.show()



def texture_segmentation(img, img_cate) :
    """
    "" Do convolution ""

    # Here, use grey as our color config
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[0], img.shape[1]

    ### Load filters
    filter_bank = scipy.io.loadmat("./HW2/Problem2/filterBank.mat")
    filters = filter_bank["F"]

    output = []
    print (filters.shape)
    for filter in np.moveaxis(filters, 2, 0) :
        filter_rotated = np.rot90(filter, 2)    # rotate 180 deg
        img_conv = convolve2d(img, filter_rotated, mode="same", boundary="symm")    # symmetric boundary
        output.append(img_conv.flatten())

    output = np.array(output)
    output = output.T
    print (output.shape)

    np.save("mountain_filtered_img.npy", output)

    """


    ### K-means
    data_fp = "mountain_filtered_img.npy" if img_cate == "m" else "zebra_filtered_img.npy"
    img_data = np.load(data_fp)
    print (img_data.shape)

    h, w = img.shape[0], img.shape[1]
    k, max_iter = 6, 1000
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)

    kmeans.fit(img_data)
    y = kmeans.predict(img_data)
    y = y.reshape(h, w)

    plt.imshow(y)
    plt.show()




def texture_segmentation2(img, img_cate) :
    h, w = img.shape[0], img.shape[1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = img.reshape(-1, 3)

    data_fp = "mountain_filtered_img.npy" if img_cate == "m" else "zebra_filtered_img.npy"
    img_data = np.load(data_fp)

    img_data = img_data.tolist()

    for i in range(len(img_data)) :
        for value in img[i] :
            img_data[i].append(value)

    img_data = np.array(img_data)

    ### K means
    k, max_iter = 6, 1000
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)

    kmeans.fit(img_data)
    y = kmeans.predict(img_data)
    y = y.reshape(h, w)

    plt.imshow(y)
    plt.show()





if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", type=str, choices=["m", "z"])
    parser.add_argument("-c", action="store", type=str, choices=["rgb", "lab"])
    parser.add_argument("-q", action="store", type=str, choices=["color", "texture", "texture2"])
    img_fp = "./HW2/Problem2/mountain.jpg" if parser.parse_args().i == "m" else "./HW2/Problem2/zebra.jpg"
    color_mode = "rgb" if parser.parse_args().c else "lab"

    img = plt.imread(img_fp)
    print ("Image size: {} x {}".format(img.shape[0], img.shape[1]))


    ### Color Segmentation
    if parser.parse_args().q == "color" :
        color_segmentation(img, color_mode)


    ### Texture Segmentation1
    elif parser.parse_args().q == "texture" :
        texture_segmentation(img, parser.parse_args().i)

    ### Texture Segmentation2
    elif parser.parse_args().q == "texture2" :
        texture_segmentation2(img, parser.parse_args().i)