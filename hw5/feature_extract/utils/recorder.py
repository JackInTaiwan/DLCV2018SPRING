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
            print (datum)
            item["lr"].append(datum["lr"])
            if datum["loss"] != None :
                item["loss"].append(datum["loss"])
            if datum["acc_train"] != None :
                item["acc_train"].append(datum["acc_train"])
            if datum["acc_test"] != None :
                item["acc_test"].append(datum["acc_test"])
            break

    else :
        print ("use new")
        new_datum = dict()
        new_datum["model_name"] = model_name
        new_datum["batch_size"] = datum["batch_size"]
        new_datum["decay"] = datum["decay"]
        new_datum["lr_init"] = datum["lr_init"]
        new_datum["lr"] = [datum["lr"]]
        new_datum["record_period"] = datum["record_period"]
        new_datum["loss"] = []
        new_datum["acc_train"] = []
        new_datum["acc_test"] = []
        data.append(new_datum)

    with open(record_fp, "w") as f :
        data_json = json.dumps(data)
        json.dump(data_json, f)
