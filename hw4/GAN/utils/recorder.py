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
<<<<<<< HEAD
            item["loss"].append(datum["KLD_loss"])
=======
            item["loss"].append(datum["recon_loss"])
            item["KLD_loss"].append(datum["KLD_loss"])
>>>>>>> 2da6e8c88ca228dcc68777f1abb3cc81d875dc51
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
        new_datum["recon_loss"] = [datum["recon_loss"]]
        new_datum["loss"] = [datum["loss"]]
        data.append(new_datum)

    with open(record_fp, "w") as f :
        data_json = json.dumps(data)
        json.dump(data_json, f)
