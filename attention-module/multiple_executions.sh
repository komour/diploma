python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-2 concurrent invert --run-name "inv-SAM-2, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number -2 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-2 concurrent --run-name "SAM-2, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 2 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-2 concurrent outer --run-name "outer-SAM-2, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 20 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-2 concurrent invert outer --run-name "inv-outer-SAM-2, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number -20

