python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "512baseline, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 0 &&
python3 train.py --arch BAM --lr 1e-4 --tags SAM-2 outer BAM concurrent --run-name "512outer-SAM-2, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 20
