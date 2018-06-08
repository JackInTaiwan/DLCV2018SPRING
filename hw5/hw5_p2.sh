wget https://www.dropbox.com/s/4bawd51d5v4702l/model_1.pkl?dl=0 -O model_1.pkl
wget https://www.dropbox.com/s/mrbcos72kt0hhob/model_3.pkl?dl=0 -O model_3.pkl
mv model_1.pkl ./CNN/models/
mv model_3.pkl ./RNN/models/

python3 ./RNN/problem.py --data $1 --label $2 --output $3 --vgg ./CNN/models/model_1.pkl --load ./RNN/models/model_3.pkl