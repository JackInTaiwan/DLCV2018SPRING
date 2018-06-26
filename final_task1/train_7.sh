echo "========== 99 =========="
python3 main.py -i 99 --version 20 --way 5 --shot 5 --lr 0.000005 --record ./records/relationnet --step 150000

echo "========== 100 =========="
python3 main.py -i 100 --version 20 --way 5 --shot 5 --lr 0.000001 --record ./records/relationnet --step 150000

echo "========== 91 =========="
python3 main.py -i 98 --version 19 --way 5 --shot 5 --lr 0.000003 --record ./records/relationnet --step 100000 --load records/relationnet/relationnet_98.json