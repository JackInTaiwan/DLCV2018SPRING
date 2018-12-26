from itertools import *
from sklearn.decomposition import PCA
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier as KNN

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys



""" Parameters """
NUM_CLASS = 40
NUM_IMAGE = 6
N_COMPONENTS = None
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 46



""" Load Data """
data = []

class_index = np.array([[i for j in range(NUM_IMAGE)] for i in range(1, NUM_CLASS + 1)]).flatten()

for (cla_index, img_index) in zip(class_index, cycle(range(1, NUM_IMAGE + 1))) :
    fp = "./hw1_dataset/{}_{}.png".format(cla_index, img_index)
    img = cv2.imread(fp)    # [numpy]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB to gray
    vector = img.flatten()
    data.append(vector)

data = np.array(data)



""" PCA """
pca = PCA(n_components=N_COMPONENTS)
pca.fit(data)



""" Question 1 """
def q1() :
    meanface = pca.mean_
    eigenface = pca.components_[:3]


    ### Plot Meanface
    meanface = meanface.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    plt.figure(0)
    plt.title("Meanface")
    plt.imshow(meanface, cmap="gray")


    ### Plot 3 Eigenfaces
    eigenface = eigenface.reshape(3, IMAGE_HEIGHT, IMAGE_WIDTH)

    for (i, face) in enumerate(eigenface) :
        plt.figure(i + 1)
        plt.title("Eigenface_{}".format(i + 1))
        plt.imshow(face, cmap="gray")

    plt.show()



""" Question 2"""
def q2() :
    fp = "./hw1_dataset/1_1.png"
    image = cv2.imread(fp)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(1, -1)
    n_components_list = [3, 50, 100, 239]
    components = pca.components_[:]

    for (i, n_components) in enumerate(n_components_list) :
        pca.components_ = components[:n_components][:]
        proj = pca.transform(image)
        recons = pca.inverse_transform(proj)
        recons_image = recons.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        mse = np.mean((recons - image) ** 2 )

        plt.figure(i)
        plt.title("n = {} \nwith MSE: {}".format(n_components, mse))
        plt.imshow(recons_image, cmap="gray")

    plt.show()



""" Question 3"""
def q3() :
    ### Training set
    n_s = [3, 50, 159]
    k_s = [1, 3, 5]

    for n in n_s :
        pca = PCA(n_components=n)
        x_train = pca.fit_transform(data[:])
        y_train = np.array([[i for j in range(NUM_IMAGE)] for i in range(NUM_CLASS)]).flatten()
        knn = KNN()

        train_scores, test_scores = validation_curve(
            estimator=knn,
            X=x_train,
            y=y_train,
            param_name="n_neighbors",
            param_range=k_s,
            cv=3
        )

        print (test_scores.mean(axis=1))


    ### Testing set
    data_test = []

    class_index = np.array([[i for j in range(4)] for i in range(1, NUM_CLASS + 1)]).flatten()

    for (cla_index, img_index) in zip(class_index, cycle(range(NUM_IMAGE + 1, 11))):
        fp = "./hw1_dataset/{}_{}.png".format(cla_index, img_index)
        img = cv2.imread(fp)  # [numpy]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB to gray
        vector = img.flatten()
        data_test.append(vector)

    best_k, best_n = 1, 50
    knn = KNN(n_neighbors=best_k)
    pca = PCA(n_components=best_n)

    data_test = np.array(data_test)
    x_train = pca.fit_transform(data[:])
    x_test = pca.transform(data_test)
    y_test = np.array([[i for j in range(4)] for i in range(NUM_CLASS)]).flatten()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print (score)



""" Main """
if __name__ == "__main__" :
    param = sys.argv[1]     # (1, 2, 3)
    if param == "1" : q1()
    elif param == "2" : q2()
    elif param == "3" : q3()
    else : print ("[Error] Parameter is wrong.")
