import numpy as np
import os

import matplotlib.pyplot as plt



def pic_to_npy(pic_dir_fp, output_fp, mode="sat", limit=float("inf")) :
    """
    This would go through all pictures in the pic_dir and save all pictures in one .npy.
    :param pic_dir: [str] the path of dir where the pictures are stored. 
    :return: None
    """

    if mode == "sat" :
        sat_pic_to_npy(pic_dir_fp, output_fp, limit)

    elif mode == "mask" :
        mask_pic_to_npy(pic_dir_fp, output_fp, limit)



def sat_pic_to_npy(pic_dir_fp, output_fp, limit) :
    output = np.array([])
    total = len(os.listdir(pic_dir_fp)) / 2
    i = 0
    for file in os.listdir(pic_dir_fp) :
        if "sat" in file :
            i += 1
            img = plt.imread(os.path.join(pic_dir_fp, file))
            print ("Num: {}/{} | File: {} | Size: {}".format(i, total, file, img.shape))

            if len(output) == 0 :
                output = np.array([img])
            else :
                print (output.shape)
                output = np.vstack((output, np.array([img])))


    np.save(output_fp, output)

    print (
        "Task is done.\n",
        "Output size: {}".format(output.shape),
        "Output saving path: {}".format(output_fp),
    )



def mask_pic_to_npy(pic_dir_fp, output_fp, limit) :
    output = np.array([])
    total = len(os.listdir(pic_dir_fp)) / 2
    for i in range(int(total)) :
        if i >= limit :
            break
        else :
            file = "{:0>4}_mask.png".format(i)
            img = plt.imread(os.path.join(pic_dir_fp, file))
            trans_v = np.array([1, 2, 4])
            # Here use transform vector
            # (255, 0, 0) -> 1
            # (255, 0, 255) -> 5

            img_encoded = img.dot(trans_v)
            # we need to modify due to the TA's error.
            # there is no 1 (red) for our encoding, {0, 2, 3, 4, 5, 6, 7} only, so we do -1
            img_encoded = img_encoded - 1 if img_encoded != 0 else 0

            print ("Num: {}/{} | File: {} | Size: {}".format(i, total, file, img.shape))

            if len(output) == 0 :
                output = np.array([img_encoded])
            else :
                output = np.vstack((output, np.array([img_encoded])))


    np.save(output_fp, output)

    print (
        "Task is done.\n",
        "Output size: {}".format(output.shape),
        "Output saving path: {}".format(output_fp),
    )