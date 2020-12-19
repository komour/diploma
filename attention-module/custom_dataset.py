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
segm_suffix = "_segmentation"
jpg = '.jpg'
png = '.png'


class DatasetISIC2018(Dataset):
    """ISIC2018 dataset."""

    def __init__(self, label_file, root_dir, perform_flips, transform=None):
        """
        Args:
            label_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.perform_flips = perform_flips
        self.image_to_onehot = {}
        self.image_names = []
        self.root_dir = root_dir
        self.transform = transform
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
                segm = TF.vflip(img)
        if self.transform:
            img = self.transform(img)
            segm = self.transform(segm)
        label = label_to_tensor(self.image_to_onehot[self.image_names[idx]])
        return {'image': img, 'label': label, 'segm': segm}

    def __len__(self):
        return len(self.image_to_onehot)
