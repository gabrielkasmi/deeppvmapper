#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EVALUATE A TRAINING RUN — metrics JSON + prediction previews (refs #11)

Evaluates a checkpoint on the requested BDAPPV split, using the same
deterministic split/subset logic as train.py (same seed, same fraction => the
same validation images), measures single-image inference FPS, and writes
input | GT | prediction previews.

  python eval.py --checkpoint runs/segformer-mini/checkpoints/best.pth

Write the PR-ready summary next to the committed previews:

  python eval.py --checkpoint runs/segformer-mini/checkpoints/best.pth \
      --out experiments/results/segformer-mini.json \
      --previews-dir experiments/results/previews/segformer-mini
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'src'))

import argparse
import json

import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from training import (
    BDAPPVSegDataset,
    SegMetrics,
    SyntheticSegDataset,
    build_model,
    list_pairs,
    measure_inference_fps,
    param_count_millions,
    save_triplet_png,
    split_from_annotations,
    split_random,
    take_subset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a BDAPPV training checkpoint (refs #11)')
    parser.add_argument('--checkpoint', required=True,
                        help='path to best.pth / last.pth from a training run')
    parser.add_argument('--config', default=None,
                        help='run config (default: <run dir>/config.yaml)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                        help='BDAPPV split to evaluate (default: val)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-previews', type=int, default=6,
                        help='prediction preview PNGs to save')
    parser.add_argument('--out', default=None,
                        help='also write the metrics JSON here '
                             '(e.g. experiments/results/segformer-mini.json)')
    parser.add_argument('--previews-dir', default=None,
                        help='where to write preview PNGs '
                             '(default: <run dir>/eval/previews)')
    parser.add_argument('--skip-fps', action='store_true',
                        help='skip the inference FPS measurement')
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = args.config or os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'rb') as f:
        cfg = yaml.safe_load(f)

    device = (args.device or cfg.get('device')
              or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA requested but unavailable — using CPU.')
        device = 'cpu'

    # ── Model ─────────────────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if cfg.get('model_spec') is None:
        # allow evaluating with the base configs/train.yaml instead of the
        # run's dumped config — resolve the spec from the model zoo
        with open(cfg.get('models_config', 'configs/model.yaml'), 'rb') as f:
            cfg['model_spec'] = yaml.safe_load(f)[cfg.get('model')]
    model = build_model(cfg['model_spec'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    params_m = param_count_millions(model)

    threshold = float(cfg.get('threshold', 0.5))
    image_size = cfg.get('image_size', 512)
    seed = cfg.get('seed', 42)

    # ── Data — same deterministic split/subset as training ────────────────────
    if cfg.get('dry_run'):
        val_set = SyntheticSegDataset(8, image_size, seed=seed + 1)
        n_images = len(val_set)
    else:
        provider = cfg['data_provider']
        images_dir = os.path.join(cfg['data_dir'], provider, 'img')
        masks_dir = os.path.join(cfg['data_dir'], provider, 'mask')
        pairs = list_pairs(images_dir, masks_dir)
        if cfg.get('split_mode', 'annotations') == 'annotations':
            annotations = os.path.join(cfg['data_dir'],
                                       cfg.get('annotations_file', 'annotations.csv'))
            split = split_from_annotations(annotations, provider, pairs)
        else:
            split = split_random(pairs, cfg.get('val_fraction', 0.2), seed)
        selected = take_subset(pairs, split, args.split,
                               cfg.get('subset', 1.0), seed)
        val_set = BDAPPVSegDataset(selected, image_size)
        n_images = len(selected)

    if n_images == 0:
        raise SystemExit('No images in the {!r} split — was the run trained '
                         'with a different split?'.format(args.split))

    batch_size = args.batch_size or int(cfg.get('batch_size', 3))
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=int(cfg.get('num_workers', 4)),
                        pin_memory=(device == 'cuda'))

    # ── Metrics + previews ────────────────────────────────────────────────────
    metrics = SegMetrics()
    previews_dir = args.previews_dir or os.path.join(run_dir, 'eval', 'previews')
    os.makedirs(previews_dir, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for images, masks, stems in tqdm.tqdm(loader,
                                              desc='eval ({})'.format(args.split),
                                              leave=False):
            logits = model(images.to(device))
            preds = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy()
            targets = masks.squeeze(1).numpy() > 0.5
            metrics.update(preds, targets)
            for i in range(images.shape[0]):
                if saved >= args.num_previews:
                    break
                save_triplet_png(images[i], targets[i], preds[i],
                                 os.path.join(previews_dir, stems[i] + '.png'))
                saved += 1

    results = metrics.compute()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'model_name': cfg.get('model'),
        'checkpoint_path': os.path.relpath(checkpoint_path).replace('\\', '/'),
        'split': args.split,
        'images': n_images,
        'params_M': params_m,
        'input_size': image_size,
        'threshold': threshold,
        'seed': seed,
        'subset': cfg.get('subset', 1.0),
    }
    for k, v in results.items():
        summary['{}_{}'.format(args.split, k)] = v  # e.g. val_mIoU, val_pixelF1

    if not args.skip_fps:
        summary['inference_fps'] = round(
            measure_inference_fps(model, device, image_size,
                                  batches=int(cfg.get('fps_batches', 50))), 2)

    # Training-side stats, when the run's metrics.json is available
    train_metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.isfile(train_metrics_path):
        with open(train_metrics_path) as f:
            train_summary = json.load(f)
        summary['epochs_trained'] = len(train_summary.get('history', []))
        summary['avg_epoch_time_s'] = train_summary.get('avg_epoch_time_s')

    print(json.dumps(summary, indent=2))

    out_path = args.out or os.path.join(run_dir, 'eval', 'metrics.json')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Written: {}'.format(out_path))


if __name__ == '__main__':
    main()
