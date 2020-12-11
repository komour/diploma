from custom_dataset import DatasetISIC2018
import torchvision.transforms as transforms
import torch
import numpy as np


def main():
    label_file = 'images-onehot.txt'
    root_dir = 'images/256ISIC2018_Task1-2_Training_Input'
    dataset = DatasetISIC2018(label_file, root_dir, transforms.ToTensor())
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample['label'], end='\n\n')
        return


if __name__ == '__main__':
    main()
