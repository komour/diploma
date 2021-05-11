import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa


def label_to_tensor(label):
    label_list = []
    for c in label[:-1]:
        label_list.append(int(c))
    return torch.Tensor(label_list)


segm_suffix = "_segmentation"
jpg = '.jpg'
png = '.png'


class DatasetISIC2018(Dataset):
    """ISIC2018 dataset."""

    def __init__(self, label_file, root_dir, segm_dir, size0=224, perform_flips=False, perform_crop=False,
                 perform_rotate=False, perform_jitter=False, perform_gaussian_noise=False, perform_iaa_augs=False,
                 transform=None):
        """
        Args:
            label_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Some additional transformations only for image
            perform_flips (bool): If true: perform same random horizontal and random vertical flips on image and mask
            perform_crop (bool): If true: perform same RandomResizedCrop(224) for image and mask
        """
        self.transform = transform
        self.size0 = size0
        self.segm_dir = segm_dir
        self.perform_crop = perform_crop
        self.perform_flips = perform_flips
        self.perform_rotate = perform_rotate
        self.perform_jitter = perform_jitter
        self.image_to_onehot = {}
        self.image_names = []
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.perform_iaa_augs = perform_iaa_augs
        self.perform_gaussian_noise = perform_gaussian_noise

        self.iaa_aug = ImgAugTransform()
        self.gaussian_noise = ImgAugGaussianNoise()

        # DEFAULT_MEAN = [0.70843003, 0.58212194, 0.53605963]
        # DEFAULT_STD = [0.15741858, 0.1656929, 0.18091279]

        # DEFAULT_MEAN = [0.70318156, 0.5638717, 0.51286779]
        # DEFAULT_STD = [0.08849113, 0.11114756, 0.12423524]
        #
        # self.normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # self.pil_images = []
        # self.pil_images_segm = []

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            name, label = line.split(' ')
            self.image_names.append(name)
            self.image_to_onehot[name] = label
            # img_path = os.path.join(self.root_dir, name + jpg)
            # segm_path = segm_dir + name + segm_suffix + png
            # img = Image.open(img_path).convert('RGB')
            # segm = Image.open(segm_path).convert('RGB')
            # self.pil_images.append(img)
            # self.pil_images_segm.append(segm)

    def __getitem__(self, idx):
        # img = self.pil_images[idx]
        # segm = self.pil_images_segm[idx]
        img_path = os.path.join(self.root_dir,
                                self.image_names[idx] + jpg)
        segm_path = self.segm_dir + self.image_names[idx] + segm_suffix + png
        img = Image.open(img_path).convert('RGB')
        segm = Image.open(segm_path).convert('RGB')

        # same random transformations for image and mask
        if self.perform_flips:
            if random.random() > 0.5:
                img = TF.hflip(img)
                segm = TF.hflip(segm)
            if random.random() > 0.5:
                img = TF.vflip(img)
                segm = TF.vflip(segm)
        if self.perform_rotate:
            img, segm = rotate_image_and_mask(img, segm)
        if self.perform_crop:
            scale = (0.08, 1.0)
            ratio = (3. / 4., 4. / 3.)
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale, ratio)
            size = (self.size0, self.size0)
            img = TF.resized_crop(img, i, j, h, w, size, Image.BILINEAR)
            segm = TF.resized_crop(segm, i, j, h, w, size, Image.NEAREST)
        if self.transform:  # only used for CenterCrop in val/test datasets
            img = self.transform(img)
            segm = self.transform(segm)

        no_norm_image = self.to_tensor(img.copy())

        if self.perform_jitter:
            if random.random() > 0.1:
                img = self.color_jitter(img)
        if self.perform_gaussian_noise:
            if random.random() > 0.1:
                img = self.gaussian_noise(img)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(segm)
        # plt.show()
        if self.perform_iaa_augs:
            if random.random() > 0.05:
                img, segm = self.iaa_aug(img, segm)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(segm)
        # plt.show()
        img = self.to_tensor(img)

        img = self.normalize(img)
        segm = self.to_tensor(segm)
        label = label_to_tensor(self.image_to_onehot[self.image_names[idx]])
        return {'image': img, 'label': label, 'segm': segm, 'name': self.image_names[idx],
                'no_norm_image': no_norm_image}

    def __len__(self):
        return len(self.image_to_onehot)


def rotate_image_and_mask(image, mask):
    r = random.random()
    if 0 <= r < 0.33:
        degrees = 90
    elif 0.33 <= r < 0.66:
        degrees = 180
    else:
        degrees = 270
    image = TF.rotate(image, degrees)
    mask = TF.rotate(mask, degrees)
    return image, mask


class ImgAugGaussianNoise:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransform:
    def __init__(self):
        self.seq_img = iaa.Sequential([
            iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, rotate=(-45, 45), order=1, cval=0, mode=["constant", "edge"], name="MyAffine")
        ])
        self.seq_mask = iaa.Sequential([
            iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, rotate=(-45, 45), order=0, cval=0, mode=["constant", "edge"], name="MyAffine")
        ])

    def __call__(self, img, mask):
        # code to perform same random augmentation on image and mask
        seq_img = self.seq_img.localize_random_state()
        seq_img_i = seq_img.to_deterministic()
        seq_mask_i = self.seq_mask.to_deterministic()

        seq_mask_i = seq_mask_i.copy_random_state(seq_img_i, matching="name")

        img = np.array(img)
        mask = np.array(mask)

        return seq_img_i.augment_image(img), seq_mask_i.augment_image(mask)
