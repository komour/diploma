python3 train.py --tags concurrent baseline checkpoint --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "checkpoint-baseline, lr = 1" --lr 1 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "checkpoint-baseline, lr = 10" --lr 10 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "checkpoint-baseline, lr = 1e3" --lr 1e3 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "checkpoint-baseline, lr = 1e4" --lr 1e4 --cuda-device 3 &&
python3 train.py --tags concurrent baseline checkpoint --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "checkpoint-baseline, lr = 1e5" --lr 1e5 --cuda-device 3
