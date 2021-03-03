python3 train.py --tags concurrent SAM-1 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1, lr = 1e-4" --lr 1e-4 --cuda-device 3 --number 0 &&
python3 train.py --tags concurrent SAM-1 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1, lr = 1e-5" --lr 1e-5 --cuda-device 3 --number 0 &&
python3 train.py --tags concurrent SAM-1 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1, lr = 1e-6" --lr 1e-6 --cuda-device 3 --number 0 &&
python3 train.py --tags concurrent SAM-4 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-4, lr = 1e-4" --lr 1e-4 --cuda-device 3 --number 1 &&
python3 train.py --tags concurrent SAM-4 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-4, lr = 1e-5" --lr 1e-5 --cuda-device 3 --number 1 &&
python3 train.py --tags concurrent SAM-4 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-4, lr = 1e-6" --lr 1e-6 --cuda-device 3 --number 1 &&
python3 train.py --tags concurrent SAM-8 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-8, lr = 1e-4" --lr 1e-4 --cuda-device 3 --number 2 &&
python3 train.py --tags concurrent SAM-8 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-8, lr = 1e-5" --lr 1e-5 --cuda-device 3 --number 2 &&
python3 train.py --tags concurrent SAM-8 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-8, lr = 1e-6" --lr 1e-6 --cuda-device 3 --number 2 &&
python3 train.py --tags concurrent SAM-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-14, lr = 1e-4" --lr 1e-4 --cuda-device 3 --number 3 &&
python3 train.py --tags concurrent SAM-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-14, lr = 1e-5" --lr 1e-5 --cuda-device 3 --number 3 &&
python3 train.py --tags concurrent SAM-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-14, lr = 1e-6" --lr 1e-6 --cuda-device 3 --number 3 &&
python3 train.py --tags concurrent SAM-1-4-8-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1-4-8-14, lr = 1e-4" --lr 1e-4 --cuda-device 3 --number 4 &&
python3 train.py --tags concurrent SAM-1-4-8-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1-4-8-14, lr = 1e-5" --lr 1e-5 --cuda-device 3 --number 4 &&
python3 train.py --tags concurrent SAM-1-4-8-14 weighted --resume "checkpoints/RESNET50_CBAM_new_name_wrap.pth" --run-name "weighted-SAM-1-4-8-14, lr = 1e-6" --lr 1e-6 --cuda-device 3 --number 4
