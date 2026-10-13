# -*- coding: utf-8 -*-

"""
Model construction for the BDAPPV backbone experiments (refs #11).

All candidates are built with segmentation-models-pytorch (smp); encoders
prefixed `tu-` come from timm. Models return raw logits (B, 1, H, W) —
apply torch.sigmoid() + threshold to get masks.
"""

import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        'segmentation-models-pytorch is required for training — '
        'pip install -r requirements-train.txt'
    ) from exc


ARCHITECTURES = {
    'segformer': smp.Segformer,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'unet': smp.Unet,
}


def build_model(spec, pretrained=True):
    """Builds a segmentation model from a configs/model.yaml entry.

    Falls back to random init (with a warning) if pretrained encoder weights
    cannot be resolved or downloaded, so quick experiments stay runnable
    offline.
    """
    arch = spec.get('arch')
    if arch not in ARCHITECTURES:
        raise ValueError('Unknown arch {!r} — available: {}'.format(
            arch, sorted(ARCHITECTURES)))

    kwargs = dict(
        encoder_name=spec.get('encoder'),
        encoder_weights=spec.get('encoder_weights') if pretrained else None,
        in_channels=spec.get('in_channels', 3),
        classes=spec.get('classes', 1),
    )
    try:
        return ARCHITECTURES[arch](**kwargs)
    except Exception as exc:
        if kwargs['encoder_weights'] is None:
            raise
        print('WARNING: could not load encoder weights ({}); '
              'continuing from random init.'.format(exc))
        kwargs['encoder_weights'] = None
        return ARCHITECTURES[arch](**kwargs)


def param_count_millions(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


class BCEDiceLoss(nn.Module):
    """0.5 * BCE + 0.5 * soft Dice — standard combo for binary segmentation."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def build_loss(name):
    if name == 'bce_dice':
        return BCEDiceLoss()
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'dice':
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    raise ValueError(
        "Unknown loss {!r} — expected 'bce_dice', 'bce' or 'dice'".format(name))
