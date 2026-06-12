# -*- coding: utf-8 -*-

"""
DETECTION

In-memory classification pipeline.

For each tile in the batch:
  1. Read the full jp2 into a numpy array (one GDAL call, no temp thumbnails)
  2. Build an InMemoryTileDataset and classify all patches in batches
  3. Save only positive patches as GeoTIFFs directly to temp/segmentation/
  4. Record results in raw_detection_results.json

This eliminates the thousands of disk writes/reads that dominated preprocessing time.
"""

import os
import json
import numpy as np
import torch
import tqdm
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from osgeo import gdal
except ImportError:
    import gdal


def _build_patch_meta(width, height, patch_size):
    """Returns list of (xOffset, yOffset, xNN, yNN) for every patch in the tile."""
    x_shifts = int(width  / patch_size) + 1
    y_shifts = int(height / patch_size) + 1
    x_max    = width  - patch_size / 2
    y_max    = height - patch_size / 2

    meta = []
    row  = -1
    for i in range(x_shifts * y_shifts):
        if i % x_shifts == 0:
            row += 1
        xNN = min(patch_size / 2 + patch_size * (i % x_shifts), x_max)
        yNN = min(patch_size / 2 + patch_size * row, y_max)
        meta.append((int(xNN - patch_size / 2), int(yNN - patch_size / 2), xNN, yNN))
    return meta


class Detection:
    """In-memory tile classification."""

    def __init__(self, configuration):
        self.temp_dir   = configuration.get('temp_dir')
        self.model_dir  = configuration.get('model_dir')
        self.device     = configuration.get('device')
        self.batch_size = configuration.get('cls_batch_size')
        self.threshold  = configuration.get('cls_threshold')
        self.patch_size = configuration.get('patch_size')
        self.model_name = configuration.get('cls_model')

    def initialization(self):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model  = torch.load(
            os.path.join(self.model_dir, self.model_name + '.pth'),
            map_location='cpu',      # load on CPU first to avoid CUDA/MPS mismatch
            weights_only=False,
        )
        model.to(device)
        model.eval()
        return model, device

    def run(self, tile_paths):
        """
        tile_paths : {tile_name: jp2_path} from PreProcessing.run()
        """
        model, device = self.initialization()

        # Import here to stay consistent with sys.path setup in main.py
        from dataset       import InMemoryTileDataset
        from tiles_processing import save_geotiff

        seg_dir  = os.path.join(self.temp_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)

        transform = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        )

        model_outputs = {}
        print('Starting classification. {} tiles to process.'.format(len(tile_paths)))

        for tile_name, jp2_path in tile_paths.items():
            print('Processing tile {}...'.format(tile_name))

            ds           = gdal.Open(jp2_path)
            geotransform = ds.GetGeoTransform()
            width        = ds.RasterXSize
            height       = ds.RasterYSize

            # Read entire tile in one shot
            tile_arr = ds.ReadAsArray()          # (bands, H, W) uint8
            ds       = None
            if tile_arr.ndim == 2:               # single-band edge case
                tile_arr = np.stack([tile_arr, tile_arr, tile_arr])

            patch_meta = _build_patch_meta(width, height, self.patch_size)
            dataset    = InMemoryTileDataset(
                tile_arr, patch_meta, self.patch_size, geotransform, transform=transform
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)

            model_outputs[tile_name] = []

            for inputs, names, indices in tqdm.tqdm(loader, desc=tile_name):
                with torch.no_grad():
                    inputs  = inputs.to(device)
                    probs   = F.softmax(model(inputs), dim=1)[:, 1]
                    pos_idx = torch.where(probs >= self.threshold)[0]

                for i in pos_idx:
                    name     = names[i]
                    orig_idx = indices[i].item()
                    xOffset, yOffset, xNN, yNN = patch_meta[orig_idx]

                    patch = tile_arr[:, yOffset:yOffset + self.patch_size,
                                      xOffset:xOffset + self.patch_size].copy()
                    # Pad if at tile boundary
                    ph = self.patch_size - patch.shape[1]
                    pw = self.patch_size - patch.shape[2]
                    if ph > 0 or pw > 0:
                        patch = np.pad(patch, [(0, 0), (0, ph), (0, pw)])

                    save_geotiff(
                        patch[0], patch[1], patch[2],
                        xNN, yNN, self.patch_size, geotransform,
                        os.path.join(seg_dir, name),
                    )
                    model_outputs[tile_name].append(name)

            del tile_arr   # free memory before next tile

        self._save_results(model_outputs)
        print('Classification complete.')

    def _save_results(self, model_outputs):
        raw_path = os.path.join(self.temp_dir, 'raw_detection_results.json')
        if not os.path.isfile(raw_path):
            with open(raw_path, 'w') as f:
                json.dump(model_outputs, f, indent=2)
        else:
            existing = json.load(open(raw_path))
            existing.update(model_outputs)
            with open(raw_path, 'w') as f:
                json.dump(existing, f, indent=2)
