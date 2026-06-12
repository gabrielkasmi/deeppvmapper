# -*- coding: utf-8 -*-

"""
SEGMENTATION

Runs the segmentation model on positive classification patches stored in
temp/segmentation/, converts binary masks to LAMB93 polygon coordinates,
merges adjacent polygons into pseudo-arrays, and exports arrays_{dpt}.geojson.
"""

import sys
sys.path.append('../src')

import os
import glob
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.utils.data import DataLoader

try:
    from osgeo import gdal
except ImportError:
    import gdal

import dataset
import tiles_processing
import data_handlers


class Segmentation:

    def __init__(self, configuration, dpt):
        self.temp_dir          = configuration.get('temp_dir')
        self.model_dir         = configuration.get('model_dir')
        self.outputs_dir       = configuration.get('outputs_dir')
        self.source_images_dir = configuration.get('source_images_dir')
        self.device            = configuration.get('device')
        self.batch_size        = configuration.get('seg_batch_size')
        self.threshold         = configuration.get('seg_threshold')
        self.num_gpu           = configuration.get('num_gpu', 1)
        self.model_name        = configuration.get('seg_model')
        self.dpt               = dpt

    # ------------------------------------------------------------------

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
        if self.num_gpu > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.num_gpu)))
            model.to(device)
        return model, device

    # ------------------------------------------------------------------

    def inference(self, model, device):
        """Runs the segmentation model; returns (binary_masks, img_names)."""
        print('Segmenting positive thumbnails...')

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225),
            ),
        ])

        dataset_dir  = os.path.join(self.temp_dir, 'segmentation')
        data_source  = dataset.BDPVSegmentationNoLabels(dataset_dir, transform=transforms)
        loader       = DataLoader(data_source, batch_size=self.batch_size)

        outputs, img_names = [], []

        for images, names in tqdm.tqdm(loader):
            with torch.no_grad():
                predicted   = model(images.to(device))
                predictions = predicted['out']
                # min-max normalisation
                pmin, pmax = torch.min(predictions), torch.max(predictions)
                predictions = (predictions - pmin) / (pmax - pmin + 1e-9)
                binary      = (predictions >= self.threshold).long().squeeze(1)
                outputs.append(binary.detach().cpu().numpy())
                img_names.append(names)

        img_names = list(sum(img_names, ()))
        return np.concatenate(outputs), img_names

    # ------------------------------------------------------------------

    def convert_to_coordinates(self, outputs, img_names):
        """Converts binary masks to LAMB93 polygon dicts sorted by tile."""
        polygons    = tiles_processing.masks_to_coordinates(
            outputs, img_names, os.path.join(self.temp_dir, 'segmentation')
        )
        coordinates = tiles_processing.sort_polygons(polygons, self.source_images_dir)

        # Append to existing raw_segmentation_results.json
        raw_path = os.path.join(self.temp_dir, 'raw_segmentation_results.json')
        if not os.path.isfile(raw_path):
            with open(raw_path, 'w') as f:
                json.dump(coordinates, f, indent=2)
        else:
            existing = json.load(open(raw_path))
            existing.update(coordinates)
            with open(raw_path, 'w') as f:
                json.dump(existing, f, indent=2)

        return coordinates

    # ------------------------------------------------------------------

    def generate_pseudo_arrays(self, coordinates):
        """Merges adjacent detection polygons and exports arrays_{dpt}.geojson."""
        print('Merging detection polygons...')
        merged = {}

        for tile in tqdm.tqdm(coordinates):
            merged[tile] = {}
            jp2 = glob.glob(
                self.source_images_dir + '/**/{}.jp2'.format(tile), recursive=True
            )
            if not jp2:
                continue
            ds = gdal.Open(jp2[0])
            merged_polys = data_handlers.aggregate_polygons(
                list(coordinates[tile].values()), ds
            )
            for idx, poly in enumerate(merged_polys):
                merged[tile][idx] = poly

        data_handlers.export_to_geojson(merged, self.dpt, self.outputs_dir)

    # ------------------------------------------------------------------

    def run(self):
        model, device = self.initialization()
        outputs, img_names = self.inference(model, device)
        coordinates = self.convert_to_coordinates(outputs, img_names)
        self.generate_pseudo_arrays(coordinates)
        # Clean up positive patches — no longer needed after segmentation
        shutil.rmtree(os.path.join(self.temp_dir, 'segmentation'))
