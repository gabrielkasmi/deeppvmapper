# -*- coding: utf-8 -*-

"""Segmentation metrics: mIoU, pixel F1, instance-level F1, inference FPS."""

import time

import numpy as np
import torch
from scipy import ndimage

INSTANCE_IOU = 0.5                    # one-to-one match threshold for instance F1
_CONN8 = np.ones((3, 3), dtype=int)   # 8-connectivity for connected components

_EPS = 1e-7


def _instance_counts(gt, pred):
    """(matched, n_gt, n_pred) for one image, greedy one-to-one IoU matching."""
    gt_labels, n_gt = ndimage.label(gt, structure=_CONN8)
    pred_labels, n_pred = ndimage.label(pred, structure=_CONN8)

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gt == 0:
        return 0, 0, n_pred
    if n_pred == 0:
        return 0, n_gt, 0

    # gt x pred intersection histogram (index 0 on each axis = background)
    combined = gt_labels.astype(np.int64) * (n_pred + 1) + pred_labels
    hist = np.bincount(
        combined.ravel(), minlength=(n_gt + 1) * (n_pred + 1)
    ).reshape(n_gt + 1, n_pred + 1)

    inter = hist[1:, 1:].astype(np.float64)
    gt_area = hist[1:, :].sum(axis=1).astype(np.float64)
    pred_area = hist[:, 1:].sum(axis=0).astype(np.float64)
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = inter / np.maximum(union, 1.0)

    # greedy one-to-one matching, best IoU first
    candidates = np.argwhere(iou >= INSTANCE_IOU)
    candidates = sorted(candidates, key=lambda g_p: -iou[g_p[0], g_p[1]])
    matched = 0
    gt_used = np.zeros(n_gt, dtype=bool)
    pred_used = np.zeros(n_pred, dtype=bool)
    for g, p in candidates:
        if not gt_used[g] and not pred_used[p]:
            gt_used[g] = True
            pred_used[p] = True
            matched += 1
    return matched, n_gt, n_pred


class SegMetrics:
    """Accumulates pixel- and instance-level statistics over batches."""

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.matched = 0
        self.gt_instances = 0
        self.pred_instances = 0
        self.images = 0

    def update(self, pred, target):
        """pred, target: bool arrays of shape (B, H, W)."""
        pred = np.asarray(pred, dtype=bool)
        target = np.asarray(target, dtype=bool)
        self.tp += float(np.logical_and(pred, target).sum())
        self.fp += float(np.logical_and(pred, ~target).sum())
        self.fn += float(np.logical_and(~pred, target).sum())
        self.tn += float(np.logical_and(~pred, ~target).sum())
        for i in range(pred.shape[0]):
            matched, n_gt, n_pred = _instance_counts(target[i], pred[i])
            self.matched += matched
            self.gt_instances += n_gt
            self.pred_instances += n_pred
            self.images += 1

    def compute(self):
        iou_pv = self.tp / (self.tp + self.fp + self.fn + _EPS)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + _EPS)
        precision = self.tp / (self.tp + self.fp + _EPS)
        recall = self.tp / (self.tp + self.fn + _EPS)
        pixel_f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn + _EPS)
        instance_f1 = (2.0 * self.matched /
                       (2.0 * self.matched
                        + (self.gt_instances - self.matched)
                        + (self.pred_instances - self.matched)
                        + _EPS))
        return {
            'mIoU': float((iou_pv + iou_bg) / 2.0),
            'iouPV': float(iou_pv),
            'iouBG': float(iou_bg),
            'pixelPrecision': float(precision),
            'pixelRecall': float(recall),
            'pixelF1': float(pixel_f1),
            'instanceF1': float(instance_f1),
            'matchedInstances': int(self.matched),
            'gtInstances': int(self.gt_instances),
            'predInstances': int(self.pred_instances),
            'images': int(self.images),
        }


@torch.no_grad()
def measure_inference_fps(model, device, image_size, batches=50, warmup=10):
    """Single-image forward-pass throughput (images/s) at image_size x image_size."""
    was_training = model.training
    model.eval()
    try:
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        for _ in range(warmup):
            model(dummy)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(batches):
            model(dummy)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return batches / elapsed
    finally:
        if was_training:
            model.train()
