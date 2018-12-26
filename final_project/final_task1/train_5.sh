echo "========== 80 =========="
python3 main.py -i 80 --version 15 --way 5 --shot 5 --lr 0.0001 --record ./records/relationnet --step 20000

echo "========== 81 =========="
python3 main.py -i 81 --version 15 --way 5 --shot 5 --lr 0.00005 --record ./records/relationnet --step 20000

echo "========== 82 =========="
python3 main.py -i 82 --version 15 --way 5 --shot 5 --lr 0.00001 --record ./records/relationnet --step 20000
