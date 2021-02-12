python3 train20epoch.py --tags concurrent baseline --run-name "baseline visualisation, lr = 0.1" --epochs 20 --lr 0.1 --cuda-device 2 &&
python3 train20epoch.py --tags concurrent baseline --run-name "baseline visualisation, lr = 1" --epochs 20 --lr 1 --cuda-device 2 &&
python3 train20epoch.py --tags concurrent baseline --run-name "baseline visualisation, lr = 100" --epochs 20 --lr 100 --cuda-device 2 &&
python3 train20epoch.py --tags concurrent baseline --run-name "baseline visualisation, lr = 1000" --epochs 20 --lr 1000 --cuda-device 2