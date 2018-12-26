# DLCV Final Project Challenge 1 Object Recognition 


### Task 1 
Run
```console
# in final/task1-small-data-supervised-learning/
bash task1_predict.sh $1 $2
# $1 test dataset dir path
# $2 output dir path
```
Example
```console
bash task1_predict.sh Fashion_MNIST_student/test/ output/
```
Then, it would produce one .csv file `predict.csv` in `output/` directory.


### Task2
Run
```console
# in final/task2-few-shot-learning/Method1-CNN_KNN/
bash final_cnn.sh $1 $2 $3
# $1 training novel dataset dir path
# $2 test dataset dir path
# $3 output dir path
```
Example
```console
bash final_cnn.sh task2-dataset/novel/ test/ output/
```
Then, it would produce three .csv files `prediction_1_shot.csv`, `prediction_5_shot.csv` and `prediction_10_shot.csv` in `output/` directory.
