python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 baseline concurrent --run-name "outer-SAM-2, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 baseline concurrent --run-name "outer-SAM-2, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM outer-SAM-2 baseline concurrent --run-name "outer-SAM-2, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&

python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 baseline concurrent --run-name "test-outer-SAM-3, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 baseline concurrent --run-name "test-outer-SAM-3, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-3 --arch BAM --tags BAM outer-SAM-3 baseline concurrent --run-name "test-outer-SAM-3, +augmentations" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2
