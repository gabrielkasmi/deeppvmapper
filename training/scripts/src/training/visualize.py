# -*- coding: utf-8 -*-

"""PNG helpers for qualitative previews (input | ground truth | prediction)."""

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def denormalize(image_tensor):
    """(3, H, W) normalized float tensor -> (H, W, 3) uint8 array."""
    arr = image_tensor.cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    return (arr * 255).round().astype(np.uint8)


def mask_to_rgb(mask_bool):
    """(H, W) bool -> (H, W, 3) uint8, mask in white."""
    rgb = np.zeros(mask_bool.shape + (3,), dtype=np.uint8)
    rgb[mask_bool] = 255
    return rgb


def save_triplet_png(image_tensor, gt_bool, pred_bool, out_path):
    """Saves input | ground truth | prediction side by side."""
    image = denormalize(image_tensor)
    gt = mask_to_rgb(np.asarray(gt_bool, dtype=bool))
    pred = mask_to_rgb(np.asarray(pred_bool, dtype=bool))
    panel = np.concatenate([image, gt, pred], axis=1)
    Image.fromarray(panel).save(out_path)
