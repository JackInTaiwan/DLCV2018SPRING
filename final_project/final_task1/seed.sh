for i in {1..100}
do
    echo "========== $i =========="
    python3 main.py -i 153 --net classifier --version 31 --trainer 4 --record records/classifier/ --load records/classifier/classifier_153.json --step 1 --lr 0.00000001 --seed $i
done
