python3 train.py --lr 1e-6 --cuda-device 3 --resume "checkpoints/1e-5seq1-outer-SAM-1_checkpoint.pth" --is-server 1 --run-name "1e-6seq1-1-SAM-1" --tags SAM-1 sequential-1-1 &&
python3 train.py --lr 1e-6 --cuda-device 3 --resume "checkpoints/1e-5seq1-SAM-1_checkpoint.pth" --is-server 1 --run-name "1e-6seq1-1-outer-SAM-1" --tags outer-SAM-1 sequential-1-1

