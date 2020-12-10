import os
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


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

    def __init__(self, label_file, root_dir, transform=None):
        """
        Args:
            label_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_to_onehot = {}
        self.image_names = []
        self.root_dir = root_dir
        self.transform = transform
        f = open(label_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            name, label = line.split(' ')
            name = name
            self.image_names.append(name)
            self.image_to_onehot[name] = label

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.image_names[idx] + jpg)
        segm_path = segm_dir + self.image_names[idx] + segm_suffix + png
        img = Image.open(img_path)
        segm = Image.open(segm_path)
        if self.transform:
            img = self.transform(img)
            segm = self.transform(segm)
        label = label_to_tensor(self.image_to_onehot[self.image_names[idx]])
        return {'image': img, 'label': label, 'segm': segm}

    def __len__(self):
        return len(self.image_to_onehot)
