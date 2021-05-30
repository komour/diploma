python3 train.py --run-type outer-SAM-all --arch ResNet18BAM --tags BAM outer concurrent --run-name "0+outer-BAM-all" --cuda-device 3 &&
python3 train.py --run-type outer-SAM-1 --arch ResNet18BAM --tags BAM outer concurrent --run-name "0+outer-BAM-1" --cuda-device 3
