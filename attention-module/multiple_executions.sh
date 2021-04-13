python3 train.py --arch BAM --lr 1e-3 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM baseline concurrent --run-name "baseline, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM inv-mask-sam --run-name "inv-mask-sam, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number -1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM inv-mask-sam --run-name "inv-mask-sam, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number -1 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM inv-mask-sam --run-name "inv-mask-sam, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number -1 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM inv-mask-sam --run-name "inv-mask-sam, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number -1 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM SAM-1 concurrent --run-name "SAM-1, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 1 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-1 concurrent --run-name "SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 1 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM SAM-1 concurrent --run-name "SAM-1, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 1 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM SAM-1 concurrent --run-name "SAM-1, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 1 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 10 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 10 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 10 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM outer SAM-1 concurrent --run-name "outer-SAM-1, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 10 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM SAM-2 concurrent --run-name "SAM-2, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 2 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-2 concurrent --run-name "SAM-2, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 2 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM SAM-2 concurrent --run-name "SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 2 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM SAM-2 concurrent --run-name "SAM-2, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 2 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 20 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 20 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 20 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM outer SAM-2 concurrent --run-name "outer-SAM-2, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 20 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM SAM-3 concurrent --run-name "SAM-3, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 3 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-3 concurrent --run-name "SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 3 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM SAM-3 concurrent --run-name "SAM-3, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 3 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM SAM-3 concurrent --run-name "SAM-3, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 3 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 30 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 30 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 30 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM outer SAM-3 concurrent --run-name "outer-SAM-3, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 30 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM SAM-all concurrent --run-name "SAM-all, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 5 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM SAM-all concurrent --run-name "SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 5 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM SAM-all concurrent --run-name "SAM-all, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 5 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM SAM-all concurrent --run-name "SAM-all, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 5 &&

python3 train.py --arch BAM --lr 1e-3 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-3" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 50 &&
python3 train.py --arch BAM --lr 1e-4 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-4" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 50 &&
python3 train.py --arch BAM --lr 1e-5 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-5" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 50 &&
python3 train.py --arch BAM --lr 1e-6 --tags BAM outer SAM-all concurrent --run-name "outer-SAM-all, lr = 1e-6" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar  --cuda-device 2 --number 50
