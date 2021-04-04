from custom_dataset import DatasetISIC2018
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import os

valdir = os.path.join('data', 'val')
val_labels = os.path.join('data', 'val', 'images_onehot_val.txt')

traindir = os.path.join('data', 'train')
train_labels = os.path.join('data', 'train', 'images_onehot_train.txt')

train_vis_file = open('vis_train.txt')
train_vis_image_names = train_vis_file.readlines()


def main():
    size0 = 224
    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        False,
        False,
        transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
    # for i, dictionary in enumerate(val_loader):
    #     input_img = dictionary['image']
    #     target = dictionary['label']
    #     print(input_img.size())

    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        True,  # perform flips
        True  # perform random resized crop
    )

    # # test_dataset = DatasetISIC2018(
    # #     test_labels,
    # #     testdir,
    # #     False,
    # #     False,
    # # )
    train_sampler = None
    #
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)
    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        name = dictionary['name'][0]
        print(f'{name}l 1')
        return 0


if __name__ == '__main__':
    # print({train_vis_image_names[0]})
    name = "ISIC_0012212"
    if name+'\n' in train_vis_image_names:
        print("YAAY")
    # print(torch.__version__)
    # main()
    # image = torch.zeros(1, 3, 224, 224)
    # maxpool = nn.MaxPool3d(kernel_size=(3, 4, 4))
    # desired = torch.zeros(1, 1, 56, 56)
    # processed_image = maxpool(image)
    # print(processed_image.size())
