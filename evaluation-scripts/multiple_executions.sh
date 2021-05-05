python3 ../train.py --run-type baseline --arch BAM --tags BAM baseline concurrent --run-name "baseline,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 ../train.py --run-type baseline --arch BAM --tags BAM baseline concurrent --run-name "baseline,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 ../train.py --run-type baseline --arch BAM --tags BAM baseline concurrent --run-name "baseline,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&

python3 ../train.py --run-type outer-SAM-1 --arch BAM --tags BAM outer-SAM-1 concurrent --run-name "outer-SAM-1,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 ../train.py --run-type outer-SAM-1 --arch BAM --tags BAM outer-SAM-1 concurrent --run-name "outer-SAM-1,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2 &&
python3 ../train.py --run-type outer-SAM-1 --arch BAM --tags BAM outer-SAM-1 concurrent --run-name "outer-SAM-1,  aug-vol.2" --resume checkpoints/RESNET50_IMAGENET_BAM_best.pth.tar --cuda-device 2
