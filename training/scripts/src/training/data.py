# -*- coding: utf-8 -*-

"""
BDAPPV data loading for segmentation training (refs #11).

Expects the raw file layout of the Hugging Face dataset gabrielkasmi/bdappv,
e.g. after:

    hf download gabrielkasmi/bdappv --repo-type dataset --local-dir data/bdappv
    (older hub versions: huggingface-cli download ...)

    data/bdappv/
    ├── ign/                  (or google/)
    │   ├── img/*.png         400x400 aerial images
    │   └── mask/*.png        binary PV masks (positives only)
    ├── annotations.csv       canonical split manifest (one row per image)
    ├── metadata.csv
    └── README.md

Segmentation trains on positives only: images that have a matching mask file.
The train/val/test assignment comes from the dataset's canonical
department-based split (annotations.csv) — do not re-split, published BDAPPV
results depend on it.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Same normalization as the pipeline's segmentation step (segmentation.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

DOWNLOAD_HINT = (
    'hf download gabrielkasmi/bdappv --repo-type dataset --local-dir data/bdappv'
)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# File listing and splits
# ---------------------------------------------------------------------------

def list_pairs(images_dir, masks_dir):
    """Sorted list of (image_path, mask_path) for positive samples.

    Positives = images with a mask file of the same stem. Negatives (images
    without a mask) are classification data, not segmentation data.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            'Images directory not found: {}\n'
            'Download the dataset first (see experiments/README.md):\n'
            '  {}'.format(images_dir, DOWNLOAD_HINT)
        )

    masks_by_stem = {}
    if os.path.isdir(masks_dir):
        for f in os.listdir(masks_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTENSIONS:
                masks_by_stem[stem] = os.path.join(masks_dir, f)

    pairs = []
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTENSIONS and stem in masks_by_stem:
            pairs.append((os.path.join(images_dir, f), masks_by_stem[stem]))
    return pairs


def split_from_annotations(annotations_csv, provider, pairs):
    """Maps image stems to their canonical split using annotations.csv.

    The manifest has one row per (installation x provider), with an `image`
    column (path or file name) and a `split` column (train/val/test).
    Returns {stem: 'train'|'val'|'test'} for the stems present in `pairs`.
    """
    if not os.path.isfile(annotations_csv):
        raise FileNotFoundError(
            'Annotations file not found: {}\n'
            'It ships with the dataset snapshot (annotations.csv at the root). '
            'See experiments/README.md, or set split_mode: random in '
            'configs/train.yaml as a fallback (breaks comparability with '
            'published results).'.format(annotations_csv)
        )

    import pandas as pd

    table = pd.read_csv(annotations_csv)
    columns = list(table.columns)
    image_col = next(
        (c for c in ('image', 'img', 'filename', 'file') if c in columns), None
    )
    split_col = next(
        (c for c in ('split', 'dataset') if c in columns), None
    )
    if image_col is None or split_col is None:
        raise ValueError(
            'Unexpected annotations.csv schema (columns: {}). Expected an '
            'image/file column and a `split` column. Inspect the CSV and '
            'adjust split_from_annotations(), or set split_mode: random.'.format(columns)
        )

    provider_prefix = provider + '/'
    stem_to_split = {}
    for name, split in zip(table[image_col], table[split_col]):
        name = str(name)
        normalized = name.replace('\\', '/')
        if '/' in normalized and not normalized.startswith(provider_prefix):
            continue  # row belongs to the other imagery provider
        stem_to_split[_stem(normalized)] = str(split)

    split = {}
    unknown = 0
    for image_path, _ in pairs:
        stem = _stem(image_path)
        if stem in stem_to_split:
            split[stem] = stem_to_split[stem]
        else:
            unknown += 1

    if unknown == len(pairs):
        raise ValueError(
            'No image matched a row of {} — the `image` column does not '
            'reference the files as this loader expects. Inspect the CSV and '
            'adjust split_from_annotations(), or set split_mode: random.'.format(annotations_csv)
        )
    if unknown:
        print('WARNING: {} images missing from annotations.csv — dropped.'.format(unknown))
    return split


def split_random(pairs, val_fraction, seed):
    """Seeded random train/val split — fallback only, NOT comparable with
    published BDAPPV results (the canonical split is department-based)."""
    stems = [_stem(p[0]) for p in pairs]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(stems))
    n_val = int(round(val_fraction * len(stems)))
    split = {}
    for i in perm[:n_val]:
        split[stems[i]] = 'val'
    for i in perm[n_val:]:
        split[stems[i]] = 'train'
    return split


def take_subset(pairs, split, wanted_split, fraction, seed):
    """Deterministically keeps `fraction` of the pairs belonging to one split.

    Same (seed, fraction, file list) => same subset, so all candidate models
    are compared on identical data regardless of when they are trained.
    """
    selected = [p for p in pairs if split.get(_stem(p[0])) == wanted_split]
    if fraction >= 1.0 or not selected:
        return selected

    n = int(round(fraction * len(selected)))
    n = max(min(n, len(selected)), 1)
    if n >= len(selected):
        return selected
    rng = np.random.RandomState(seed)
    keep = set(rng.permutation(len(selected))[:n].tolist())
    return [p for i, p in enumerate(selected) if i in keep]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _augment(image, mask, augmentations):
    """Joint geometric augmentations — image and mask stay aligned."""
    if 'rot90' in augmentations:
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
    if 'hflip' in augmentations and np.random.rand() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]
    if 'vflip' in augmentations and np.random.rand() < 0.5:
        image = image[::-1]
        mask = mask[::-1]
    return image, mask


class BDAPPVSegDataset(Dataset):
    """Positive BDAPPV pairs -> (image, mask, stem), resized to image_size.

    image: float32 (3, S, S), ImageNet-normalized
    mask : float32 (1, S, S), 0/1
    """

    def __init__(self, pairs, image_size=512, augmentations=(),
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.pairs = pairs
        self.image_size = image_size
        self.augmentations = set(augmentations)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0  # HWC
        mask = np.asarray(mask) > 127                        # HW bool

        if self.augmentations:
            image, mask = _augment(image, mask, self.augmentations)

        image = image.transpose(2, 0, 1)                     # CHW
        image = (image - self.mean) / self.std
        mask = mask.astype(np.float32)[None]                 # (1, H, W)

        return (torch.from_numpy(image.copy()),
                torch.from_numpy(mask.copy()),
                _stem(image_path))


class SyntheticSegDataset(Dataset):
    """Random images + blob masks — used by train.py --dry-run (no dataset,
    no network: encoder weights are skipped, device defaults to CPU)."""

    def __init__(self, n_samples, image_size=512, seed=0,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)  # deterministic per sample
        image = rng.randint(0, 256, size=(self.image_size, self.image_size, 3))
        image = image.astype(np.float32) / 255.0
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        for _ in range(rng.randint(1, 5)):
            y = rng.randint(0, self.image_size - 64)
            x = rng.randint(0, self.image_size - 64)
            mask[y:y + rng.randint(16, 96), x:x + rng.randint(16, 96)] = True

        image = (image.transpose(2, 0, 1) - self.mean) / self.std
        mask = mask.astype(np.float32)[None]
        return (torch.from_numpy(image),
                torch.from_numpy(mask),
                'synthetic-{:04d}'.format(idx))
