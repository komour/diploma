python3 train.py --data-type ISIC256 --workers 2 --lr 1e-5 --run-name "baseline, lr=1e-5" --run-type baseline --arch ResNet18BAM --tags ResNet18BAM baseline concurrent --cuda-device 2 &&
python3 train.py --data-type ISIC256 --workers 2 --lr 1e-5 --run-name "outer-all, lr=1e-5" --run-type outer-SAM-all --arch ResNet18BAM --tags ResNet18BAM outer-SAM-all concurrent --cuda-device 2 &&
python3 train.py --data-type ISIC256 --workers 2 --lr 1e-5 --run-name "outer-1, lr=1e-5" --run-type outer-SAM-1 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-1 concurrent --cuda-device 2 &&
python3 train.py --data-type ISIC256 --workers 2 --lr 1e-5 --run-name "outer-3, lr=1e-5" --run-type outer-SAM-3 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-3 concurrent --cuda-device 2 &&
python3 train.py --data-type ISIC256 --workers 2 --lr 1e-5 --run-name "outer-2, lr=1e-5" --run-type outer-SAM-2 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-2 concurrent --cuda-device 2