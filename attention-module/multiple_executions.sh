python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 &&

python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 10 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 10 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 10 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 10 &&

python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 20 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 20 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 20 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 20 &&

python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 30 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 30 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 30 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 30 &&

python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 50 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 50 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 50 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 1 --number 50