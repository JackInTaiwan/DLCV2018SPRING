import scipy.io
import argparse
import cv2
import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier as KNN



def sub1() :
    img = cv2.imread("./HW2/Problem3/train-10/Coast/image_0006.jpg")
    s = cv2.xfeatures2d.SIFT_create()
    kps = s.detect(img)
    kpcs = s.compute(img, kps)
    print (kpcs[1].shape)
    
    for k in kps:
        cv2.circle(img, (int(k.pt[0]), int(k.pt[1])), 1, (0, 255, 0), -1)
    plt.imshow(img)
    plt.show()



def sub2() :
    """
    data = np.array([])

    fn = "./HW2/Problem3/train-10"
    for cate_dir in os.listdir(fn) :
        for file in os.listdir(fn + "/" + cate_dir) :
            fp_img = fn + "/" + cate_dir + "/" + file
            img = cv2.imread(fp_img)
            s = cv2.xfeatures2d.SIFT_create()
            kps = s.detect(img)
            descs = s.compute(img, kps)
            descs = descs[1]
            if len(data) == 0 :
                data = descs
            else :
                data = np.vstack((data, descs))

    print (data)
    print (data.shape)
    np.save("./q3_2_descriptors.npy", data)
    """

    data = np.load("./q3_2_descriptors.npy")
    k, max_iter = 50, 5000
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans.fit(data)
    y = kmeans.predict(data)



    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    #np.save("./q3_2_pca.npy", data_pca)


    fig = plt.figure()
    ax = Axes3D(fig)

    #data_pca = np.load("./q3_2_pca.npy")

    cmap = ["blue", "red", "yellow", "#fa23a0", "#a4ff3a", "#3490fa"]
    for i, cluster in enumerate(y) :
        if cluster in range(6) :
            ax.scatter(data_pca[i][0], data_pca[i][1], data_pca[i][2], c=cmap[cluster], alpha=0.3)

    for i, center in enumerate(kmeans.cluster_centers_[:6]) :
        x = pca.transform([center])
        ax.scatter(x[0][0], x[0][1], x[0][2], c=cmap[i], marker="*", s=300)

    plt.show()



def sub3(strategy) :
    ### K means
    """
    data = np.load("./q3_2_descriptors.npy")
    k, max_iter = 50, 5000
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans.fit(data)
    np.save("./q3_3_vwords.npy", kmeans.cluster_centers_)
    """

    vwords = np.load("./q3_3_vwords.npy")

    fn = "./HW2/Problem3/train-10"
    for cate_dir in os.listdir(fn) :
        file = os.listdir(fn + "/" + cate_dir)[0]
        fp_img = fn + "/" + cate_dir + "/" + file
        img = cv2.imread(fp_img)

        s = cv2.xfeatures2d.SIFT_create()
        kps = s.detect(img)
        kpcs = s.compute(img, kps)
        kpcs = kpcs[1]

        table = []

        for kpc in kpcs :
            dis = [distance.euclidean(kpc, vwords[i]) for i in range(len(vwords))]
            table.append(dis)

        table = np.array(table)

        ### Hard-sum strategy
        if strategy == "hs" :
            table_hard_sum = np.argmin(table, axis=1)

            plt.hist(table_hard_sum, bins=np.array(range(50)))
            plt.show()


        ### Soft-sum strategy
        elif strategy == "ss" :
            for i in range(len(table)) :
                for j in range(len(table[i])) :
                    table[i][j] = 1 / table[i][j]
                table[i] = table[i] / sum(table[i])

            table = np.sum(table, axis=0) / table.shape[0]
            plt.bar(range(50), table)
            plt.show()

        ### Soft-max strategy
        elif strategy == "sm" :
            for i in range(len(table)) :
                for j in range(len(table[i])) :
                    table[i][j] = 1 / table[i][j]
                table[i] = table[i] / sum(table[i])
            table_soft_max = np.max(table, axis=0)
            plt.bar(range(50), table_soft_max)
            plt.show()



def sub4(strategy) :

    def get_data() :
        ### Training data
        #vwords = np.load("./q3_3_vwords.npy")
        vwords = np.load("./q4_4_vwords_100.npy")

        #fn = "./HW2/Problem3/train-10"
        #fn = "./HW2/Problem3/train-100"
        fn = "./HW2/Problem3/test-100"

        x_train, y_train = np.array([]), []


        for cate_index, cate_dir in enumerate(os.listdir(fn)) :
            print ("cate", cate_dir)
            for file_index, file in enumerate(os.listdir(fn + "/" + cate_dir)) :
                print (file_index)
                fp_img = fn + "/" + cate_dir + "/" + file
                img = cv2.imread(fp_img)

                s = cv2.xfeatures2d.SIFT_create()
                kps = s.detect(img)
                kpcs = s.compute(img, kps)
                kpcs = kpcs[1]


                ### Table for one picture
                table = []

                for kpc in kpcs :
                    dis = [distance.euclidean(kpc, vwords[i]) for i in range(len(vwords))]
                    table.append(dis)

                table = np.array(table)


                ### Hard-sum strategy
                if strategy == "hs" :
                    table_hard_sum = np.argmin(table, axis=1)
                    bow = table_hard_sum.flatten()
                    bow = np.array([np.count_nonzero(bow == i) for i in range(50)]) / len(bow)

                    if len(x_train) == 0 :
                        x_train = np.array([bow])
                    else :
                        x_train = np.vstack((x_train, bow))

                    y_train.append(cate_index)


                ### Soft-sum strategy
                elif strategy == "ss" :
                    for i in range(len(table)) :
                        for j in range(len(table[i])) :
                            table[i][j] = 1 / table[i][j]
                        table[i] = table[i] / sum(table[i])

                    table = np.sum(table, axis=0) / table.shape[0]
                    bow = table.flatten()

                    if len(x_train) == 0 :
                        x_train = np.array([bow])
                    else :
                        x_train = np.vstack((x_train, bow))

                    y_train.append(cate_index)


                ### Soft-max strategy
                elif strategy == "sm" :
                    for i in range(len(table)):
                        for j in range(len(table[i])):
                            table[i][j] = 1 / table[i][j]
                        table[i] = table[i] / sum(table[i])
                    table_soft_max = np.max(table, axis=0)
                    bow = table_soft_max.flatten()

                    if len(x_train) == 0 :
                        x_train = np.array([bow])
                    else :
                        x_train = np.vstack((x_train, bow))

                    y_train.append(cate_index)

        x_train, y_train = x_train, np.array(y_train)
        print (x_train.shape)
        print (y_train.shape)

        #np.savez("x_train_bow_{}.npz".format(strategy), x_train=x_train, y_train =y_train)
        #np.savez("x_train_100_bow_{}.npz".format(strategy), x_train=x_train, y_train=y_train)
        #np.savez("x_test_bow_{}.npz".format(strategy), x_test=x_train, y_test=y_train)
        np.savez("x_test_100_bow_{}.npz".format(strategy), x_test=x_train, y_test=y_train)


    def get_vwords_100() :
        data = np.array([])

        fn = "./HW2/Problem3/train-100"
        for cate_dir in os.listdir(fn):
            for file in os.listdir(fn + "/" + cate_dir):
                fp_img = fn + "/" + cate_dir + "/" + file
                img = cv2.imread(fp_img)
                s = cv2.xfeatures2d.SIFT_create()
                kps = s.detect(img)
                descs = s.compute(img, kps)
                descs = descs[1]
                if len(data) == 0:
                    data = descs
                else:
                    data = np.vstack((data, descs))

        print(data)
        print(data.shape)

        k, max_iter = 50, 2000
        kmeans = KMeans(n_clusters=k, max_iter=max_iter)
        kmeans.fit(data)
        vwords_100 = kmeans.cluster_centers_
        print (vwords_100.shape)
        np.save("./q4_4_vwords_100.npy", vwords_100)


    ### Testing
    def test() :
        #train_fn = "x_train_bow_{}.npz".format(strategy)
        train_fn = "x_train_100_bow_{}.npz".format(strategy)
        x_train, y_train = np.load(train_fn)["x_train"], np.load(train_fn)["y_train"]

        test_fn = "x_test_100_bow_{}.npz".format(strategy)
        x_test, y_test = np.load(test_fn)["x_test"], np.load(test_fn)["y_test"]

        k_s = [5, 10, 20, 30, 50]
        for k in k_s :
            #k = 5
            knn = KNN(n_neighbors=k)
            knn.fit(x_train, y_train)
            score = knn.score(x_test, y_test)

            print ("Accuracy on {} with k={}: {}".format(strategy, k, score))

    #get_data()
    #get_vwords_100()
    test()



if __name__ == "__main__" :
    #sub1()
    #sub2()

    strategy = "sm"
    #sub3(strategy)
    sub4(strategy)