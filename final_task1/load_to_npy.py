import os
import numpy as np
from matplotlib import pyplot as plt




def store_base_train():  # 80 classes, 500 images for each
    print('Store train...')
    classes = ['task2-dataset/base/' + name + '/train/' for name in os.listdir('task2-dataset/base/')]
    classes.sort()  # Image paths for each class
    base_train = np.zeros((80, 500, 32, 32, 3), dtype=np.float32)

    for class_id, path in enumerate(classes):
        img_names = [path + name for name in os.listdir(path)]
        img_names.sort()

        for img_id, img_name in enumerate(img_names):
            base_train[class_id, img_id] = plt.imread(img_name)

    np.save('base_train.npy', base_train)



def store_base_valid():  # 80 classes, 100 images for each
    print('Store valid...')
    classes = ['task2-dataset/base/' + name + '/test/' for name in os.listdir('task2-dataset/base/')]
    classes.sort()  # Image paths for each class
    base_valid = np.zeros((80, 100, 32, 32, 3), dtype=np.float32)

    for class_id, path in enumerate(classes):
        img_names = [path + name for name in os.listdir(path)]
        img_names.sort()

        for img_id, img_name in enumerate(img_names):
            base_valid[class_id, img_id] = mpimg.imread(img_name)

    np.save('base_valid.npy', base_valid)



def store_novel():  # 20 classes, 500 images for each
    print('Store novel...')
    classes = ['novel/' + name + '/train/' for name in os.listdir('novel/')]
    classes.sort()  # Image paths for each class
    novel = np.zeros((20, 500, 32, 32, 3), dtype=np.float32)

    for class_id, path in enumerate(classes):
        img_names = [path + name for name in os.listdir(path)]
        img_names.sort()

        for img_id, img_name in enumerate(img_names):
            novel[class_id, img_id] = mpimg.imread(img_name)

    np.save('novel.npy', novel)



def store_test():  # total 2000 images of novel class
    print('Store test...')
    test = np.zeros((2000, 32, 32, 3), dtype=np.float32)
    img_names = ['test/' + str(i) + '.png' for i in range(2000)]
    for img_id, img_name in enumerate(img_names):
        test[img_id] = mpimg.imread(img_name)

    np.save('test.npy', test)



def test():
    print('test 1')




if __name__ == '__main__':
    print('========= Preprocessing... ========== ')
    #store_base_train()
    store_base_valid()
    # store_novel()
    #store_test()
