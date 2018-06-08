wget https://www.dropbox.com/s/fl6v6itcj2ubx5g/model_2.pkl?dl=0 -O model_2.pkl
mv model_2.pkl ./CNN/models/
python3 ./CNN/problem.py --data $1 --label $2 --output $3 --load ./CNN/models/model_2.pkl