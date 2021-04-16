python3 train.py --lmbd 2 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent --run-name "lmbd=2, SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 3 &&
python3 train.py --lmbd 5 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent --run-name "lmbd=5, SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 3 &&
python3 train.py --lmbd 10 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent --run-name "lmbd=10, SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 3 &&

python3 train.py --lmbd 2 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent outer --run-name "lmbd=2, outer-SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 30 &&
python3 train.py --lmbd 5 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent outer --run-name "lmbd=5, outer-SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 30 &&
python3 train.py --lmbd 10 --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent outer --run-name "lmbd=10, outer-SAM-3, lr=1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 --number 30

