import json
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser




def plot(json_fp) :
    with open(json_fp) as f :
        data = json.load(f)
    data = json.loads(data)
    data = data[0]

    loss_period = 100
    loss = data["loss"]

    plt.figure(0)
    plt.title("Loss")
    plt.plot(np.array(range(len(loss))) * loss_period, loss, c="r", linewidth=1)
    plt.xlabel("Steps")
    plt.ylabel("Loss")


    acc_period = 500
    #acc_train = data["acc_train"]
    acc_test = data["acc_test"]
    acc_test_new = []
    for item in acc_test :
        acc, total = 0, 0
        for a in item :
            acc += a[0] * a[1][1]
            total += a[1][1]
        acc_test_new.append(acc / total)


    plt.figure(1)
    plt.title("Accuracy")
    #plt.plot(np.array(range(len(acc_train))) * acc_period, acc_train, c="r", linewidth=1)
    plt.plot(np.array(range(len(acc_test_new))) * acc_period, acc_test_new, c="b", linewidth=1)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    #plt.legend(["Train", "Valid"])

    plt.show()




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("--json", type=str, required=True)

    json_fp = parser.parse_args().json

    plot(json_fp)