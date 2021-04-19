python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-1 concurrent invert --run-name "inv-SAM-1, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number -1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-1 concurrent --run-name "SAM-1, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-1 concurrent outer --run-name "outer-SAM-1, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 10 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent outer --run-name "baseline, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 0

