import os
from torch.utils.data.dataset import Dataset
import torch
from skimage import io


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
        jpg = '.jpg'
        for line in lines:
            name, label = line.split(' ')
            name = name + jpg
            self.image_names.append(name)
            self.image_to_onehot[name] = label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.image_names[idx])
        image = io.imread(img_name)
        label = self.image_to_onehot[self.image_names[idx]]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_to_onehot)
