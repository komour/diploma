import os
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


def label_to_tensor(label):
    label_list = []
    for c in label[:-1]:
        label_list.append(int(c))
    return torch.Tensor(label_list)


segm_dir = "images/256ISIC2018_Task1_Training_GroundTruth/"
segm_dir = "images/512ISIC2018_Task1_Training_GroundTruth/"
segm_suffix = "_segmentation"
jpg = '.jpg'
png = '.png'


class DatasetISIC2018(Dataset):
    """ISIC2018 dataset."""

    def __init__(self, label_file, root_dir, perform_flips=False, perform_crop=False, transform=None):
        """
        Args:
            label_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Some additional transformations only for image
            perform_flips (bool): If true: perform same random horizontal and random vertical flips on image and mask
            perform_crop (bool): If true: perform same RandomResizedCrop(224) for image and mask
        """
        self.transform = transform
        self.perform_crop = perform_crop
        self.perform_flips = perform_flips
        self.image_to_onehot = {}
        self.image_names = []
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()

        # DEFAULT_MEAN = [0.70843003, 0.58212194, 0.53605963]
        # DEFAULT_STD = [0.15741858, 0.1656929, 0.18091279]

        # DEFAULT_MEAN = [0.70318156, 0.5638717, 0.51286779]
        # DEFAULT_STD = [0.08849113, 0.11114756, 0.12423524]
        #
        # self.normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.pil_images = []
        self.pil_images_segm = []

        f = open(label_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            name, label = line.split(' ')
            self.image_names.append(name)
            self.image_to_onehot[name] = label
            img_path = os.path.join(self.root_dir, name + jpg)
            segm_path = segm_dir + name + segm_suffix + png
            img = Image.open(img_path).convert('RGB')
            segm = Image.open(segm_path).convert('RGB')
            self.pil_images.append(img)
            self.pil_images_segm.append(segm)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root_dir,
        #                         self.image_names[idx] + jpg)
        # segm_path = segm_dir + self.image_names[idx] + segm_suffix + png
        # img = Image.open(img_path).convert('RGB')
        # segm = Image.open(segm_path).convert('RGB')
        img = self.pil_images[idx]
        segm = self.pil_images_segm[idx]
        if self.perform_flips:  # same random transformations for image and mask
            if random.random() > 0.5:
                img = TF.hflip(img)
                segm = TF.hflip(segm)
            if random.random() > 0.5:
                img = TF.vflip(img)
                segm = TF.vflip(segm)
        if self.perform_crop:
            scale = (0.08, 1.0)  # for conda
            # scale = [0.08, 1.0]  # for pip
            ratio = (3. / 4., 4. / 3.)  # for conda
            # ratio = [3. / 4., 4. / 3.]  # for pip
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale, ratio)
            size0 = 224
            size0 = 448
            # size = (size0, size0)  # for conda
            size = [size0, size0]  # for pip
            img = TF.resized_crop(img, i, j, h, w, size, Image.BILINEAR)  # for conda
            # img = TF.resized_crop(img, i, j, h, w, size, TF.InterpolationMode.BILINEAR)  # for pip
            segm = TF.resized_crop(segm, i, j, h, w, size, Image.NEAREST)  # for conda
            # segm = TF.resized_crop(segm, i, j, h, w, size, TF.InterpolationMode.NEAREST)  # for pip
        if self.transform:
            img = self.transform(img)
            segm = self.transform(segm)
        img = self.to_tensor(img)
        no_norm_image = img.detach().clone()
        img = self.normalize(img)
        segm = self.to_tensor(segm)
        label = label_to_tensor(self.image_to_onehot[self.image_names[idx]])
        return {'image': img, 'label': label, 'segm': segm, 'name': self.image_names[idx],
                'no_norm_image': no_norm_image}

    def __len__(self):
        return len(self.image_to_onehot)
