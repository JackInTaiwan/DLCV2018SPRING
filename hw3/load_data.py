import sys, os
#sys.path.append(os.path.abspath(".."))
from argparse import ArgumentParser
from utils import pic_to_npy



""" Parameters """
TRAIN_PICTURES_FP = "./hw3-train-validation/train"
OUTPUT_DATA_FP = "./data"



if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-x", action="store", type=str, default=False, choices=["train", "test", "val"])
    parser.add_argument("-y", action="store", type=str, default=False, choices=["train", "test", "val"])
    parser.add_argument("-f", action="store", type=str)
    parser.add_argument("-l", action="store", type=int)


    print ("======== Starting converting pictures ========")
    if parser.parse_args().x :
        name = "x_{}".format(parser.parse_args().x)
        x_fp = os.path.join(OUTPUT_DATA_FP, name)
        pic_to_npy(parser.parse_args().f, x_fp, mode="sat")

    elif parser.parse_args().y :
        name = "y_{}".format(parser.parse_args().y)
        y_fp = os.path.join(OUTPUT_DATA_FP, name)
        if parser.parse_args().l != None :
            pic_to_npy(parser.parse_args().f, y_fp, mode="mask", limit=parser.parse_args().l)
        else :
            pic_to_npy(parser.parse_args().f, y_fp, mode="mask")