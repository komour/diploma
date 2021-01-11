from custom_dataset import DatasetISIC2018
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn


def main():
    label_file = 'data/train/images_onehot_train.txt'
    root_dir = 'data/train'
    dataset = DatasetISIC2018(label_file, root_dir, transforms.ToTensor())
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample['image'].size, end='\n')


if __name__ == '__main__':
    main()
    # image = torch.zeros(1, 3, 224, 224)
    # maxpool = nn.MaxPool3d(kernel_size=(3, 4, 4))
    # desired = torch.zeros(1, 1, 56, 56)
    # processed_image = maxpool(image)
    # print(processed_image.size())
