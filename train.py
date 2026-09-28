#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TRAINING SCAFFOLD — candidate segmentation backbones on BDAPPV (refs #11)

Trains one of the candidate architectures from configs/model.yaml on the
BDAPPV positives (images with masks) and writes everything to runs/<run_name>/:

  runs/<run_name>/config.yaml     effective config used for the run
  runs/<run_name>/metrics.json    per-epoch metrics + summary
  runs/<run_name>/checkpoints/    best.pth / last.pth (never committed)
  runs/<run_name>/val_preds/      input | GT | prediction preview PNGs

Quick run (5% subset, 3 epochs — see scripts/run_quick.sh):

  python train.py --model segformer --subset 0.05 --run-name segformer-mini

Smoke test without the dataset (synthetic data, CPU, no downloads):

  python train.py --dry-run
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'src'))

import argparse
import json
import random
import time

import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from training import (
    BDAPPVSegDataset,
    SegMetrics,
    SyntheticSegDataset,
    build_loss,
    build_model,
    list_pairs,
    param_count_millions,
    save_triplet_png,
    split_from_annotations,
    split_random,
    take_subset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a candidate segmentation backbone on BDAPPV (refs #11)')
    parser.add_argument('--config', default='configs/train.yaml',
                        help='training config (default: configs/train.yaml)')
    parser.add_argument('--model', default=None,
                        help='key from configs/model.yaml (default: config value)')
    parser.add_argument('--run-name', default=None,
                        help='output dir under runs/ (default: model name)')
    parser.add_argument('--subset', type=float, default=None,
                        help='fraction of each split to use, e.g. 0.05')
    parser.add_argument('--device', default=None,
                        help='cuda | cpu (default: config value, CPU fallback)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: config value)')
    parser.add_argument('--dry-run', action='store_true',
                        help='synthetic data, 1 epoch, CPU — checks the plumbing '
                             'without the dataset or pretrained downloads')
    return parser.parse_args()


def resolve_device(cfg_device, requested, dry_run):
    if requested:
        device = requested
    elif dry_run:
        device = 'cpu'
    else:
        device = cfg_device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA requested but unavailable — using CPU.')
        device = 'cpu'
    return device


def _numpy_worker_init(worker_id):
    # torch re-seeds its own RNG per DataLoader worker but not numpy's —
    # without this, every worker would apply identical augmentations.
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def _build_optimizer(model, cfg):
    name = cfg.get('optimizer', 'adamw')
    lr = float(cfg.get('lr', 1e-4))
    weight_decay = float(cfg.get('weight_decay', 0.01))
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=float(cfg.get('sgd_momentum', 0.9)),
                               weight_decay=weight_decay)
    raise ValueError("optimizer must be 'adamw' or 'sgd', got {!r}".format(name))


def build_split_pairs(cfg, seed):
    """Returns (train_pairs, val_pairs) following the canonical BDAPPV split."""
    provider = cfg['data_provider']
    images_dir = os.path.join(cfg['data_dir'], provider, 'img')
    masks_dir = os.path.join(cfg['data_dir'], provider, 'mask')
    pairs = list_pairs(images_dir, masks_dir)
    if not pairs:
        raise RuntimeError(
            'No image/mask pairs found under {} — is the dataset downloaded? '
            'See experiments/README.md.'.format(cfg['data_dir']))

    if cfg.get('split_mode', 'annotations') == 'annotations':
        annotations = os.path.join(cfg['data_dir'],
                                   cfg.get('annotations_file', 'annotations.csv'))
        split = split_from_annotations(annotations, provider, pairs)
    elif cfg['split_mode'] == 'random':
        print('WARNING: random split — NOT comparable with published BDAPPV '
              'results (the canonical split is department-based).')
        split = split_random(pairs, cfg.get('val_fraction', 0.2), seed)
    else:
        raise ValueError("split_mode must be 'annotations' or 'random', got {!r}"
                         .format(cfg['split_mode']))

    fraction = cfg.get('subset', 1.0)
    train_pairs = take_subset(pairs, split, 'train', fraction, seed)
    val_pairs = take_subset(pairs, split, 'val', fraction, seed)
    print('Data: {} train / {} val images (subset={} of {} positives)'.format(
        len(train_pairs), len(val_pairs), fraction, len(pairs)))
    if not train_pairs or not val_pairs:
        raise RuntimeError('Empty split — train: {}, val: {}'
                           .format(len(train_pairs), len(val_pairs)))
    return train_pairs, val_pairs


def evaluate(model, loader, device, threshold):
    """Returns the SegMetrics dict for a val/test loader."""
    model.eval()
    metrics = SegMetrics()
    with torch.no_grad():
        for images, masks, _ in loader:
            logits = model(images.to(device))
            preds = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy()
            targets = masks.squeeze(1).numpy() > 0.5
            metrics.update(preds, targets)
    return metrics.compute()


def _checkpoint(model_name, cfg, model, epoch, val_metrics):
    # Plain dict (state_dict + model spec) — no full-model pickle, easy to
    # consume from the inference-side loader (#7).
    return {
        'model_name': model_name,
        'model_spec': cfg['model_spec'],
        'threshold': float(cfg.get('threshold', 0.5)),
        'epoch': epoch,
        'val_metrics': val_metrics,
        'state_dict': model.state_dict(),
    }


def _save_val_previews(model, run_dir, val_loader, device, threshold, num_previews):
    """Prediction previews from the best checkpoint (input | GT | pred)."""
    best_path = os.path.join(run_dir, 'checkpoints', 'best.pth')
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location='cpu')['state_dict'])
        model.to(device)
    model.eval()
    previews_dir = os.path.join(run_dir, 'val_preds')
    saved = 0
    with torch.no_grad():
        for images, masks, stems in val_loader:
            logits = model(images.to(device))
            preds = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy()
            gts = masks.squeeze(1).numpy() > 0.5
            for i in range(images.shape[0]):
                if saved >= num_previews:
                    return
                save_triplet_png(images[i], gts[i], preds[i],
                                 os.path.join(previews_dir, stems[i] + '.png'))
                saved += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, 'rb') as f:
        cfg = yaml.safe_load(f)

    # ── Effective config (CLI overrides) ──────────────────────────────────────
    if args.seed is not None:
        cfg['seed'] = args.seed
    seed = cfg.get('seed', 42)
    if args.subset is not None:
        cfg['subset'] = args.subset
    cfg['dry_run'] = args.dry_run

    model_name = args.model or cfg.get('model') or 'segformer'
    cfg['model'] = model_name
    with open(cfg.get('models_config', 'configs/model.yaml'), 'rb') as f:
        model_specs = yaml.safe_load(f)
    if model_name not in model_specs:
        raise SystemExit('Unknown model {!r} — available: {}'
                         .format(model_name, sorted(model_specs)))
    cfg['model_spec'] = model_specs[model_name]

    run_name = args.run_name or cfg.get('run_name') or model_name
    cfg['run_name'] = run_name
    run_dir = os.path.join('runs', run_name)
    if os.path.isfile(os.path.join(run_dir, 'config.yaml')):
        raise SystemExit('runs/{}/ already contains a config.yaml — pick another '
                         '--run-name or delete the directory.'.format(run_name))
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'val_preds'), exist_ok=True)

    device = resolve_device(cfg.get('device'), args.device, args.dry_run)
    cfg['device'] = device

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    image_size = cfg.get('image_size', 512)
    if args.dry_run:
        print('DRY RUN: synthetic data, 1 epoch, no pretrained weights.')
        cfg['epochs'] = 1
        cfg['num_workers'] = 0
        train_set = SyntheticSegDataset(16, image_size, seed=seed)
        val_set = SyntheticSegDataset(8, image_size, seed=seed + 1)
    else:
        train_pairs, val_pairs = build_split_pairs(cfg, seed)
        train_set = BDAPPVSegDataset(train_pairs, image_size,
                                     cfg.get('augmentations') or [])
        val_set = BDAPPVSegDataset(val_pairs, image_size)

    batch_size = int(cfg.get('batch_size', 3))
    num_workers = int(cfg.get('num_workers', 4))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(device == 'cuda'),
                              worker_init_fn=_numpy_worker_init)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(device == 'cuda'))

    # ── Model / loss / optimizer ──────────────────────────────────────────────
    model = build_model(cfg['model_spec'],
                        pretrained=not args.dry_run and cfg.get('pretrained', True))
    model.to(device)
    params_m = param_count_millions(model)

    criterion = build_loss(cfg.get('loss', 'bce_dice'))
    optimizer = _build_optimizer(model, cfg)

    threshold = float(cfg.get('threshold', 0.5))
    selection_metric = cfg.get('selection_metric', 'val_mIoU')
    epochs = int(cfg.get('epochs', 3))

    # Persist the effective config before training starts
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print('Model: {} ({}M params) | device: {} | epochs: {} | batch: {}'.format(
        model_name, params_m, device, epochs, batch_size))

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    epoch_times = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        running_loss, n_batches = 0.0, 0
        for images, masks, _ in tqdm.tqdm(train_loader,
                                          desc='epoch {}/{} train'.format(epoch, epochs),
                                          leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_metrics = {'val_' + k: v for k, v in
                       evaluate(model, val_loader, device, threshold).items()}
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        entry = {'epoch': epoch, 'train_loss': round(train_loss, 6),
                 'epoch_time_s': round(epoch_time, 2)}
        entry.update(val_metrics)
        history.append(entry)
        print('epoch {}/{} — train_loss {:.4f} | {} {:.4f} | val_pixelF1 {:.4f} | {:.1f}s'
              .format(epoch, epochs, train_loss, selection_metric,
                      entry.get(selection_metric, float('nan')),
                      entry['val_pixelF1'], epoch_time))

        ckpt = _checkpoint(model_name, cfg, model, epoch, val_metrics)
        torch.save(ckpt, os.path.join(run_dir, 'checkpoints', 'last.pth'))
        if entry.get(selection_metric, -1.0) > best_score:
            best_score = entry.get(selection_metric, -1.0)
            best_epoch = epoch
            torch.save(ckpt, os.path.join(run_dir, 'checkpoints', 'best.pth'))

    # ── Summary ───────────────────────────────────────────────────────────────
    best_entry = next((h for h in history if h['epoch'] == best_epoch), history[-1])
    summary = {
        'run_name': run_name,
        'model': model_name,
        'params_M': params_m,
        'image_size': image_size,
        'subset': cfg.get('subset', 1.0),
        'seed': seed,
        'device': device,
        'dry_run': args.dry_run,
        'history': history,
        'best_epoch': best_epoch,
        'best': best_entry,
        'avg_epoch_time_s': round(sum(epoch_times) / len(epoch_times), 2),
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    _save_val_previews(model, run_dir, val_loader, device, threshold,
                       int(cfg.get('num_val_previews', 8)))

    print('Done. Metrics: {}'.format(os.path.join(run_dir, 'metrics.json')))


if __name__ == '__main__':
    main()
