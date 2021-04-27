import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from custom_dataset import DatasetISIC2018

valdir = os.path.join('data', 'val')
val_labels = os.path.join('data', 'val', 'images_onehot_val.txt')

traindir = os.path.join('data', 'train')
train_labels = os.path.join('data', 'train', 'images_onehot_train.txt')

testdir = os.path.join('data', 'test')
test_labels = os.path.join('data', 'test', 'images_onehot_test.txt')

train_vis_file = open('vis_train.txt')
train_vis_image_names = train_vis_file.readlines()


def main():
    size0 = 224
    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        False,
        False
        # transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        False,  # perform flips
        False  # perform random resized crop
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    test_dataset = DatasetISIC2018(
        test_labels,
        testdir,
        False,  # perform flips
        False  # perform random resized crop
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    means = np.zeros(shape=(1, 3), dtype=float)
    stds = np.zeros(shape=(1, 3), dtype=float)
    # for i, dictionary in enumerate(train_loader):
    #     input_img = dictionary['no_norm_image']
    #     means += np.mean(input_img.numpy(), axis=(2, 3))
    #     stds += np.std(input_img.numpy(), axis=(2, 3))
    #     if i % 100 == 0:
    #         print(i)

    # for i, dictionary in enumerate(val_loader):
    #     input_img = dictionary['no_norm_image']
    #     means += np.mean(input_img.numpy(), axis=(2, 3))
    #     stds += np.std(input_img.numpy(), axis=(2, 3))
    #     if i % 100 == 0:
    #         print(i)

    for i, dictionary in enumerate(test_loader):
        input_img = dictionary['image']
        no_norm_image = dictionary['no_norm_image']

        # means += np.mean(no_norm_image.numpy(), axis=(2, 3))
        # stds += np.std(no_norm_image.numpy(), axis=(2, 3))
        if i % 100 == 0:
            print(i)

            np_input2 = np.moveaxis(torch.squeeze(no_norm_image).detach().numpy(), 0, -1)
            np_input = np.moveaxis(torch.squeeze(input_img).detach().numpy(), 0, -1)

            plt.imshow(np_input)
            plt.show()
            plt.imshow(np_input2)
            plt.show()

    AMOUNT = len(test_loader) + len(train_loader) + len(val_loader)
    print(means / AMOUNT)
    print(stds / AMOUNT)

    # mine all:
    # [[0.71006687 0.57303179 0.52221987]]
    # [[0.08687312 0.10900028 0.12315174]]

    # train:
    # [[0.71385493 0.57568106 0.52478219]]
    # [[0.08677858 0.10842204 0.12241225]]

    # val:
    # [[0.70513929 0.57603742 0.52585846]]
    # [[0.08484857 0.10812451 0.12450073]]

    # test:
    # [[0.70318156 0.5638717  0.51286779]]
    # [[0.08849113 0.11114756 0.12423524]]

    # foreign:
    # DEFAULT_MEAN = [0.70843003, 0.58212194, 0.53605963]
    # DEFAULT_STD = [0.15741858, 0.1656929, 0.18091279]


if __name__ == '__main__':
    main()
    # print({train_vis_image_names[0]})
    # name = "ISIC_0012212"
    # if name+'\n' in train_vis_image_names:
    #     print("YAAY")
    # print(torch.__version__)
    # image = torch.zeros(1, 3, 224, 224)
    # maxpool = nn.MaxPool3d(kernel_size=(3, 4, 4))
    # desired = torch.zeros(1, 1, 56, 56)
    # processed_image = maxpool(image)
    # print(processed_image.size())
