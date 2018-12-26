import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("--json", type=str, required=True)
json_fp = parser.parse_args().json
with open(json_fp, "r") as f :
    data = json.load(f)
    data = json.loads(data)
plt.figure(0)
acc_list = data["data"]["acc"]
plt.title("Novel class accuracy")
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.plot([item[0] for item in acc_list], [item[1] for item in acc_list], c="red")

plt.figure(1)
loss_list = data["data"]["loss"]
plt.xlabel("steps")
plt.ylabel("loss")
plt.plot([item[0] for item in loss_list], [item[1] for item in loss_list], c="red")

plt.show()
