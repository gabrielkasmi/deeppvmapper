# -*- coding: utf-8 -*-

# Libraries
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF



# Class for the BDPV classification dataset.
# Classification because the labels are binary
# Indicating whether or not the image contains an array.
# The images contained in the dataset are RBG images.

class BDPVClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        """
        Args:
            annotations_file (string): name of the csv file with labels
            img_dir (string): directory with all the images.
            transform (callable, optional): optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(annotations_file, header = None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        label = self.img_labels.iloc[idx, 1]
        name = self.img_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label, name

    # From here : custom attributes

    def image_name(self, idx):
        return self.img_labels.iloc[idx, 0]
    
    def labels(self):
        self.img_labels.columns = ["img_name", "label"]
        return self.img_labels

class BDPVSegmentationDataset(Dataset):
    def __init__(self, masks_dir, img_dir, image_transform = None, random_transform = False, extension = '.png'):
        """
        Args:
            annotations_file (string): directory with the labels
            img_dir (string): directory with all the images.
            image_transform (callable, optional): optional image transforms to be applied on the image.
            random_transform : indicates whether random cropping, flipping, etc should be applied to both
            the image and the mask
            extension (callable, optional): the format of the image files
            
            
            segmentation dataset is formatted as follows : 
            train/
              - img.extension
            train_masks/
              - mask.extension
              
            test/
            test_masks/
            
            masks and images have the same name to ease retrieval
        """
        self.masks_dir = masks_dir
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.random_transform = random_transform
        self.extension = extension

    def __len__(self):      
        # the length is computed as the number of masks in the masks directory
        return int(len(os.listdir(self.masks_dir)))

    def __getitem__(self, idx):
        
        masks = os.listdir(self.masks_dir)
        target = masks[idx]
        mask_path = os.path.join(self.masks_dir, target)
        img_path = os.path.join(self.img_dir, target)
        name = target[:-len(self.extension)] # remove the extension from the name
        
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        mask = transforms.ToTensor()(Image.open(mask_path))

        if self.image_transform:
            image = self.image_transform(image)

        if self.random_transform:

            # applies a series of transform to the image and the mask

            # horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # rotations (90, -90, 180, -180)
            if random.random() > 0.5:
                image = TF.rotate(image, 90.)
                mask = TF.rotate(mask, 90.)

            if random.random() > 0.5:
                image = TF.rotate(image, -90.)
                mask = TF.rotate(mask, -90.)

            if random.random() > 0.5:
                image = TF.rotate(image, 180.)
                mask = TF.rotate(mask, 180.)

            if random.random() > 0.5:
                image = TF.rotate(image, -180.)
                mask = TF.rotate(mask, -180.)
            
        return image, mask, name