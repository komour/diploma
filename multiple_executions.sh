python3 train.py --run-type baseline --arch ResNet18BAM --tags ResNet18BAM baseline concurrent --run-name "baseline,  ResNet18BAM" --cuda-device 1 &&
python3 train.py --run-type outer-SAM-all --arch ResNet18BAM --tags ResNet18BAM outer-SAM-all concurrent --run-name "outer-SAM-all,  ResNet18BAM" --cuda-device 1 &&
python3 train.py --run-type outer-SAM-1 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-1 concurrent --run-name "outer-SAM-1,  ResNet18BAM" --cuda-device 1
