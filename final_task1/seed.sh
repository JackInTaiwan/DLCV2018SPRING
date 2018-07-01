for i in {41..100}
do
    echo "========== $i =========="
    python3 main.py -i 140 --net classifier --version 31 --trainer 3 --record records/classifier/ --load records/classifier/classifier_140.json --step 100000 --lr 0.0 --seed $i --step 1
done
