import sys, os
#sys.path.append(os.path.abspath(".."))
from argparse import ArgumentParser
try :
    from .utils import pic_to_npy
except :
    from utils import pic_to_npy


""" Parameters """
TRAIN_PICTURES_FP = "./hw3-train-validation/train"
OUTPUT_DATA_FP = "./data"



if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-m", action="store", type=str, default=False, required=True, choices=["train", "test"])
    parser.add_argument("-s", action="store", type=int, default=0)
    parser.add_argument("-f", action="store", type=str)
    parser.add_argument("-l", action="store", type=int)


    print ("======== Starting converting pictures ========")
    name = "{}_data".format(parser.parse_args().m)
    x_fp = os.path.join(OUTPUT_DATA_FP, name)
    if parser.parse_args().l != None :
        pic_to_npy(parser.parse_args().f, x_fp, parser.parse_args().s, limit=parser.parse_args().l)
    else :
        pic_to_npy(parser.parse_args().f, x_fp, parser.parse_args().s)