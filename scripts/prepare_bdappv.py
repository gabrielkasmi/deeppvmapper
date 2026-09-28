#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PREPARE THE BDAPPV SNAPSHOT FOR TRAINING (refs #11)

The current Hub release of gabrielkasmi/bdappv ships Parquet shards
(`ign/train-0000N-of-000NN.parquet`, `ign/validation-*`, `ign/test-*`) whose
`image` and `mask` columns carry the PNG bytes inline — there are no `img/`
and `mask/` directories and no `annotations.csv` in the release.

training/data.py reads the raw layout, so this script materializes it once:

    data/bdappv/
    ├── ign/img/<stem>.png      extracted positives
    ├── ign/mask/<stem>.png     binary PV masks
    └── annotations.csv         image,split,provider,identifiant

Only positives (rows with a mask) are extracted — segmentation trains on
images that have a mask, and list_pairs() ignores the rest. The canonical
department-based split is taken from the shards' `split` column and written
into annotations.csv, with `validation` normalized to `val` to match the
train/val/test keys the rest of the scaffold uses.

Idempotent: re-run after downloading more shards; existing PNGs are skipped
and CSV rows are rebuilt per provider.

  python scripts/prepare_bdappv.py --provider ign
  python scripts/prepare_bdappv.py --provider ign --include-negatives
"""

import argparse
import csv
import glob
import io
import os
import sys

import pyarrow.parquet as pq
from PIL import Image

SPLIT_ALIASES = {'validation': 'val', 'valid': 'val'}
CSV_COLUMNS = ['image', 'split', 'provider', 'identifiant']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract the raw img/mask layout from the BDAPPV parquet '
                    'shards (refs #11)')
    parser.add_argument('--data-dir', default='data/bdappv',
                        help='dataset snapshot dir (default: data/bdappv)')
    parser.add_argument('--provider', default='ign',
                        help='provider subdirectory to process (default: ign)')
    parser.add_argument('--include-negatives', action='store_true',
                        help='also write images that have no mask '
                             '(classification data — unused by train.py)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='rows per parquet batch (default: 64)')
    return parser.parse_args()


def normalize_split(value):
    value = str(value).strip().lower()
    return SPLIT_ALIASES.get(value, value)


def _struct_bytes(value):
    """Parquet image/mask cells are {bytes, path} structs (or None)."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        return value.get('bytes'), value.get('path')
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    return None, None


def _stem(path, identifiant, fallback):
    if path:
        name = os.path.basename(str(path).replace('\\', '/'))
        stem = os.path.splitext(name)[0]
        if stem:
            return stem
    if identifiant:
        return str(identifiant)
    return fallback


def _write_png(raw, dest, mode):
    """Decodes `raw` and re-encodes to PNG so the extension matches the
    contract in training/data.py (IMAGE_EXTENSIONS)."""
    with Image.open(io.BytesIO(raw)) as img:
        img.convert(mode).save(dest, format='PNG')


def process_shard(shard, img_dir, mask_dir, provider, include_negatives,
                  batch_size, seen, rows):
    parquet = pq.ParquetFile(shard)
    written = skipped = duplicates = 0
    row_no = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for record in batch.to_pylist():
            row_no += 1
            fallback = '{}-{:06d}'.format(os.path.basename(shard), row_no)
            image_raw, image_path = _struct_bytes(record.get('image'))
            mask_raw, _ = _struct_bytes(record.get('mask'))
            has_mask = bool(record.get('has_mask')) and mask_raw is not None

            if image_raw is None:
                continue
            if not has_mask and not include_negatives:
                continue

            stem = _stem(image_path, record.get('identifiant'), fallback)
            if stem in seen:
                duplicates += 1
                continue
            seen.add(stem)

            image_dest = os.path.join(img_dir, stem + '.png')
            if not os.path.isfile(image_dest):
                _write_png(image_raw, image_dest, 'RGB')
                written += 1
            else:
                skipped += 1

            if has_mask:
                mask_dest = os.path.join(mask_dir, stem + '.png')
                if not os.path.isfile(mask_dest):
                    _write_png(mask_raw, mask_dest, 'L')

            rows.append({
                'image': '{}/img/{}.png'.format(provider, stem),
                'split': normalize_split(record.get('split', 'train')),
                'provider': provider,
                'identifiant': record.get('identifiant', ''),
            })

    print('  {}: {} written, {} already present, {} duplicate stems'.format(
        os.path.basename(shard), written, skipped, duplicates))
    return written


def write_annotations(annotations_path, provider, rows):
    """Rebuilds annotations.csv, preserving rows of other providers."""
    existing = []
    if os.path.isfile(annotations_path):
        with open(annotations_path, newline='') as f:
            existing = [r for r in csv.DictReader(f)
                        if r.get('provider') != provider]

    with open(annotations_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in existing + rows:
            writer.writerow({c: row.get(c, '') for c in CSV_COLUMNS})


def main():
    args = parse_args()
    provider_dir = os.path.join(args.data_dir, args.provider)

    shards = sorted(glob.glob(os.path.join(provider_dir, '*.parquet')))
    if not shards:
        raise SystemExit(
            'No parquet shards under {} — download the snapshot first:\n'
            '  hf download gabrielkasmi/bdappv --repo-type dataset '
            '--include "{}/train-*" "{}/validation-*" '
            '--local-dir {}'.format(provider_dir, args.provider,
                                    args.provider, args.data_dir))

    img_dir = os.path.join(provider_dir, 'img')
    mask_dir = os.path.join(provider_dir, 'mask')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print('Extracting {} shard(s) from {} into img/ + mask/'.format(
        len(shards), provider_dir))

    seen = set()
    rows = []
    total = 0
    for shard in shards:
        total += process_shard(shard, img_dir, mask_dir, args.provider,
                               args.include_negatives, args.batch_size,
                               seen, rows)

    annotations_path = os.path.join(args.data_dir, 'annotations.csv')
    write_annotations(annotations_path, args.provider, rows)

    counts = {}
    for row in rows:
        counts[row['split']] = counts.get(row['split'], 0) + 1
    print('Positives: {} total | {}'.format(
        len(rows), ', '.join('{} {}'.format(k, counts[k])
                             for k in sorted(counts))))
    print('Wrote {} images and {}'.format(total, annotations_path))


if __name__ == '__main__':
    main()
