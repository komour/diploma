python3 train.py --run-type baseline --arch ResNet18BAM --tags BAM baseline concurrent --run-name "cbam-sam1" --cuda-device 1 &&
python3 train.py --run-type outer-SAM-1 --arch ResNet18BAM --tags BAM outer concurrent --run-name "cbam-sam1" --cuda-device 1
