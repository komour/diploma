python3 train.py --run-type outer-SAM-all --arch ResNet18BAM --tags ResNet18BAM outer-SAM-all concurrent --run-name "outer-SAM-all,  ResNet18BAM" --cuda-device 1 &&
python3 train.py --run-type outer-SAM-2 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-2 concurrent --run-name "outer-SAM-2,  ResNet18" --cuda-device 1 &&
python3 train.py --run-type outer-SAM-3 --arch ResNet18BAM --tags ResNet18BAM outer-SAM-3 concurrent --run-name "outer-SAM-3,  ResNet18" --cuda-device 1
