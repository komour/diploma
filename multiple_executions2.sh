python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 concurrent --run-name "outer-SAM-2, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 concurrent --run-name "outer-SAM-2, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 concurrent --run-name "outer-SAM-2, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&

python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 concurrent --run-name "outer-SAM-3, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 concurrent --run-name "outer-SAM-3, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 concurrent --run-name "outer-SAM-3, aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2
