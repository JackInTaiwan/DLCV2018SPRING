import json
from argparse import ArgumentParser



if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-j", "--json", action="store", type=str, required=True, help="json file path")
    parser.add_argument("-m", "--model", action="store", type=str, required=True, help="model name")

    with open(parser.parse_args().j, "r") as f :
        data = json.load(f)

    data = json.loads(data)
    for item in data :
        if item["model_name"] == parser.parse_args().m :
            print (item)