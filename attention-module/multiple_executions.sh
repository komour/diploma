python3 train.py --tags concurrent baseline --run-name "baseline, lr = 1e-3" --lr 1e-3 --cuda-device 2 &&
python3 train.py --tags concurrent baseline --run-name "baseline, lr = 1e-4" --lr 1e-4 --cuda-device 2 &&
python3 train.py --tags concurrent baseline --run-name "baseline, lr = 1e-5" --lr 1e-5 --cuda-device 2 &&
python3 train.py --tags concurrent baseline --run-name "baseline, lr = 1e-6" --lr 1e-6 --cuda-device 2
