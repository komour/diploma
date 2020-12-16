#python train_imagenet.py \
#			--ngpu 8 \
#			--workers 20 \
#			--arch resnet --depth 50 \
#			--epochs 100 \
#			--batch-size 256 --lr 0.1 \
#			--att-type CBAM \
#			--prefix RESNET50_IMAGENET_CBAM \
#			./data/ImageNet/
#
#


python train_imagenet.py --ngpu 1 --workers 1 --arch resnet --depth 50 --epochs 100 --batch-size 2 --lr 0.1 --att-type CBAM --prefix ISIC2018_CBAM data/

# params
#			--ngpu 1 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM data/ISIC2018_10/

python3 train_imagenet_cuda.py --ngpu 4 --workers 4 --arch resnet --depth 50 --epochs 100 --batch-size 1 --lr 0.1 --att-type CBAM --prefix ISIC2018_CBAM data/