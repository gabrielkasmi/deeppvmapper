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
import time
import numpy as np
import torch
import tqdm
import torchvision
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F

try:
    from osgeo import gdal
except ImportError:
    import gdal


def _read_tile(jp2_path):
    """
    Opens and fully decodes a jp2 tile into memory.

    Runs in a background thread so the next tile can be read/decoded while
    the GPU is busy classifying the current one (see Detection.run).
    """
    ds           = gdal.Open(jp2_path)
    geotransform = ds.GetGeoTransform()
    width        = ds.RasterXSize
    height       = ds.RasterYSize
    tile_arr     = ds.ReadAsArray()          # the expensive part: JP2 decode
    ds           = None
    if tile_arr.ndim == 2:                   # single-band edge case
        tile_arr = np.stack([tile_arr, tile_arr, tile_arr])
    return tile_arr, geotransform, width, height


def _prepare_batch(dataset, idx_batch):
    """
    Builds one classification batch (stacked tensor + names + original indices).

    Runs in a background thread so batch b+1's slicing/padding/normalize work
    (pure CPU/numpy) overlaps with the GPU forward pass on batch b, instead of
    happening strictly before it (see Detection.run).
    """
    images, names, indices = [], [], []
    for idx in idx_batch:
        image, name, orig_idx = dataset[idx]
        images.append(image)
        names.append(name)
        indices.append(orig_idx)
    return torch.stack(images), names, indices


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

        tile_names = list(tile_paths.keys())
        jp2_paths  = list(tile_paths.values())

        # Prefetch: while the GPU classifies tile i, a background thread reads
        # and decodes tile i+1, so the (slow) JP2 decode of N+1 overlaps with
        # the classification of N instead of happening back-to-back.
        with ThreadPoolExecutor(max_workers=1) as tile_executor, \
             ThreadPoolExecutor(max_workers=1) as batch_executor:

            pending_tile = tile_executor.submit(_read_tile, jp2_paths[0]) if jp2_paths else None

            for i, tile_name in enumerate(tile_names):
                print('Processing tile {}...'.format(tile_name))

                # Time spent blocked here is decode time NOT hidden by prefetch
                # (0 for tile 0, since there is no previous tile to overlap with).
                t0 = time.perf_counter()
                tile_arr, geotransform, width, height = pending_tile.result()
                decode_wait_s = time.perf_counter() - t0

                if i + 1 < len(jp2_paths):
                    pending_tile = tile_executor.submit(_read_tile, jp2_paths[i + 1])

                patch_meta = _build_patch_meta(width, height, self.patch_size)
                dataset    = InMemoryTileDataset(
                    tile_arr, patch_meta, self.patch_size, geotransform, transform=transform
                )

                n_patches      = len(dataset)
                batch_idx_list = [
                    list(range(b, min(b + self.batch_size, n_patches)))
                    for b in range(0, n_patches, self.batch_size)
                ]

                model_outputs[tile_name] = []

                # Prefetch: while the GPU classifies batch b, a background thread
                # slices/pads/normalizes batch b+1 from tile_arr, so CPU prep overlaps
                # with GPU compute instead of happening strictly before it.
                pending_batch = (
                    batch_executor.submit(_prepare_batch, dataset, batch_idx_list[0])
                    if batch_idx_list else None
                )

                t1 = time.perf_counter()
                for b, idx_batch in enumerate(tqdm.tqdm(batch_idx_list, desc=tile_name)):
                    inputs, patch_names, indices = pending_batch.result()

                    if b + 1 < len(batch_idx_list):
                        pending_batch = batch_executor.submit(
                            _prepare_batch, dataset, batch_idx_list[b + 1]
                        )

                    with torch.no_grad():
                        inputs  = inputs.to(device)
                        probs   = F.softmax(model(inputs), dim=1)[:, 1]
                        # Single GPU->CPU sync for the whole batch, instead of one
                        # sync per positive patch (see Detection.run docstring history).
                        pos_idx = torch.where(probs >= self.threshold)[0].cpu().tolist()

                    for j in pos_idx:
                        name     = patch_names[j]
                        orig_idx = indices[j]
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

                classify_s = time.perf_counter() - t1
                hidden = '' if i == 0 else (
                    ' [fully hidden by prefetch]' if decode_wait_s < 1
                    else ' [decode NOT fully hidden -> decode is the bottleneck]'
                )
                print('  decode_wait={:.1f}s, classify={:.1f}s{}'.format(
                    decode_wait_s, classify_s, hidden
                ))

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
