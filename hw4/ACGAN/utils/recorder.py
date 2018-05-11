import json


def console(string) :
    print("======== {} ========".format(string))



def record(record_fp, datum) :
    model_name = datum["model_name"]

    with open(record_fp, "r") as f:
        data = json.load(f)
    data = json.loads(data)

    for item in data :
        if item["model_name"] == model_name :
            item["lr"].append(datum["lr"])
            item["loss_real"].append(datum["loss_real"])
            item["loss_fake"].append(datum["loss_fake"])
            item["acc_true"].append(datum["acc_true"])
            item["acc_false"].append(datum["acc_false"])
            break

    else :
        new_datum = dict()
        new_datum["model_name"] = model_name
        new_datum["data_size"] = datum["data_size"]
        new_datum["batch_size"] = datum["batch_size"]
        new_datum["decay"] = datum["decay"]
        new_datum["lr_init"] = datum["lr_init"]
        new_datum["lr"] = [datum["lr"]]
        new_datum["record_epoch"] = datum["record_epoch"]
        new_datum["loss_real"] = [datum["loss_real"]]
        new_datum["loss_fake"] = [datum["loss_fake"]]
        new_datum["acc_true"] = [datum["acc_true"]]
        new_datum["acc_false"] = [datum["acc_false"]]
        data.append(new_datum)

    with open(record_fp, "w") as f :
        data_json = json.dumps(data)
        json.dump(data_json, f)
