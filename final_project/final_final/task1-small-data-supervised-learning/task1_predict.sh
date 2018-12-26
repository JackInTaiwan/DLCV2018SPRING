#!/bin/bash

wget -O 2conv.h5 https://www.dropbox.com/s/r2mlif88vu8wz58/2conv.h5?dl=1
#1 test directory #2 csv output
python3 task1_predict.py $1 $2 

