# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class BDPVDatasetNoLabels(Dataset):
    """
    Generic label-free dataset for classification and segmentation inference.
    Reads GeoTIFF thumbnails from a directory.
    """

    def __init__(self, img_dir, transform=None, extension='.tif'):
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(extension)]
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        name  = self.img_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, name


# Aliases for backward compatibility
BDPVClassificationNoLabels = BDPVDatasetNoLabels
BDPVSegmentationNoLabels   = BDPVDatasetNoLabels


class InMemoryTileDataset(Dataset):
    """
    Label-free dataset built from a tile numpy array — no disk I/O during classification.

    tile_arr    : np.ndarray (3, H, W) uint8
    patch_meta  : list of (xOffset, yOffset, xNN, yNN) — one entry per patch
    patch_size  : int
    geotransform: GDAL geotransform tuple (ulx, xres, 0, uly, 0, yres)
    transform   : applied after [0,1] float conversion (typically Normalize)

    Returns (image_tensor, name, idx) where:
      - name  : geo-encoded filename matching save_geotiff output
      - idx   : original patch index for recovering patch_meta on positives
    """

    def __init__(self, tile_arr, patch_meta, patch_size, geotransform, transform=None):
        self.tile_arr   = tile_arr
        self.patch_meta = patch_meta
        self.patch_size = patch_size
        ulx, xres, _, uly, _, yres = geotransform
        self.ulx, self.xres = ulx, xres
        self.uly, self.yres = uly, yres
        self.transform = transform

    def __len__(self):
        return len(self.patch_meta)

    def __getitem__(self, idx):
        xOffset, yOffset, xNN, yNN = self.patch_meta[idx]
        patch = self.tile_arr[:, yOffset:yOffset + self.patch_size,
                               xOffset:xOffset + self.patch_size].copy()

        # Pad edge patches to full patch_size
        ph = self.patch_size - patch.shape[1]
        pw = self.patch_size - patch.shape[2]
        if ph > 0 or pw > 0:
            patch = np.pad(patch, [(0, 0), (0, ph), (0, pw)])

        # Name encodes geo-centre, identical to legacy save_geotiff naming
        x_geo = self.ulx + xNN * self.xres
        y_geo = self.uly + yNN * self.yres
        name  = f"{x_geo}-{y_geo}.tif"

        image = torch.from_numpy(patch.astype(np.float32) / 255.0)
        if self.transform:
            image = self.transform(image)

        return image, name, idx
