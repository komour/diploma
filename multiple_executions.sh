python3 train.py --run-type baseline --arch BAM --tags BAM baseline concurrent --run-name "test-baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --run-type outer-SAM-2 --arch BAM --tags BAM baseline concurrent --run-name "test-outer-SAM-2, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2
