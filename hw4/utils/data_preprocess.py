import numpy as np
import os

import matplotlib.pyplot as plt



def pic_to_npy(pic_dir_fp, output_fp, start, limit=float("inf")) :
    output = np.array([])
    total = len(os.listdir(pic_dir_fp))

    for index in range(int(total)):
        if index + start >= limit or index + start >= total:
            break

        else:
            index = index + start
            file = "{:0>5}.png".format(index)
            img = plt.imread(os.path.join(pic_dir_fp, file))
            print("Num: {}/{} | File: {} | Size: {}".format(index, total, file, img.shape))

            if len(output) == 0:
                output = np.array([img])

            else:
                print(output.shape)
                output = np.vstack((output, np.array([img])))

    np.save(output_fp, output)

    print(
        "Task is done.\n",
        "Output size: {}".format(output.shape),
        "Output saving path: {}".format(output_fp),
    )

