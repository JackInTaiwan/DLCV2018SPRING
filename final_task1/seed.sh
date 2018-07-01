<<<<<<< HEAD
for i in {10..10}
=======
for i in {1..100}
>>>>>>> d68776e223989e6e15f63e74f77095790be0201a
do
    echo "========== $i =========="
    python3 main.py -i 153 --net classifier --version 31 --trainer 4 --record records/classifier/ --load records/classifier/classifier_153.json --step 100000 --lr 0.00000001 --seed $i
done
