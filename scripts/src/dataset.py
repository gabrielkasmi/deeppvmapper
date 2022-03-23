# -*- coding: utf-8 -*-

# Libraries
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision


# Custom BDPV classification 
# This does not require a label file to access the files.


class BDPVClassificationNoLabels(Dataset):
    def __init__(self, img_dir, transform = None, extension = '.tif'):
        """
        Args:
            img_dir (string): directory with all the images.
            transform (callable, optional): optional transform to be applied on a sample.
            extension (str, optional): the extension of the files to look for.
        """
        self.img_names = [item for item in os.listdir(img_dir) if item[-4:] == extension]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = torchvision.transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # label = self.img_labels.iloc[idx, 1]
        name = self.img_names[idx]

        if self.transform:
            image = self.transform(image)

        return image, name

# Custom BDPV segmentation
# Doesn't need segmentation masks to access the files.

class BDPVSegmentationNoLabels(Dataset):

    def __init__(self, img_dir, transform = None, extension = '.tif'):
            """
            Args:
                img_dir (string): directory with all the images.
                transform (callable, optional): optional transform to be applied on a sample.
                extension (str, optional): the extension of the files to look for.
            """
            self.img_names = [item for item in os.listdir(img_dir) if item[-4:] == extension]
            self.img_dir = img_dir
            self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = torchvision.transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # label = self.img_labels.iloc[idx, 1]
        name = self.img_names[idx]

        if self.transform:
            image = self.transform(image)

        return image, name