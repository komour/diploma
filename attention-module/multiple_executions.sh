python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 0.1" --lr 0.1 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 0.01" --lr 0.01 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 1e-3" --lr 1e-3 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 1e-4" --lr 1e-4 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 1e-5" --lr 1e-5 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --run-name "checkpoint-baseline, lr = 1e-6" --lr 1e-6 --cuda-device 3
