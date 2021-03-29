python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/outer-SAM-1_checkpoint.pth" --is-server 1 --run-name "seq1-outer-SAM-1" --tags outer-SAM-1 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/outer-SAM-4_checkpoint.pth" --is-server 1 --run-name "seq1-outer-SAM-4" --tags outer-SAM-4 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/outer-SAM-8_checkpoint.pth" --is-server 1 --run-name "seq1-outer-SAM-8" --tags outer-SAM-8 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/outer-SAM-14_checkpoint.pth" --is-server 1 --run-name "seq1-outer-SAM-14" --tags outer-SAM-14 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/outer-SAM-1-4-8-14_checkpoint.pth" --is-server 1 --run-name "seq1-outer-SAM-1-4-8-14" --tags outer-SAM-1-4-8-14 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/SAM-1_checkpoint.pth" --is-server 1 --run-name "seq1-SAM-1" --tags SAM-1 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/SAM-4_checkpoint.pth" --is-server 1 --run-name "seq1-SAM-4" --tags SAM-4 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/SAM-8_checkpoint.pth" --is-server 1 --run-name "seq1-SAM-8" --tags SAM-8 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/SAM-14_checkpoint.pth" --is-server 1 --run-name "seq1-SAM-14" --tags SAM-14 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/SAM-1-4-8-14_checkpoint.pth" --is-server 1 --run-name "seq1-SAM-1-4-8-14" --tags SAM-1-4-8-14 sequential-1 &&
python3 train.py --lr 1e-4 --cuda-device 3 --resume "checkpoints/baseline_checkpoint.pth" --is-server 1 --run-name "seq1-baseline" --tags baseline
