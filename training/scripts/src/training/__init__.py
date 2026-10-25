# -*- coding: utf-8 -*-

"""Shared building blocks for the BDAPPV segmentation training scaffold (refs #11)."""

from .data import (
    BDAPPVSegDataset,
    SyntheticSegDataset,
    list_pairs,
    split_from_annotations,
    split_random,
    take_subset,
)
from .metrics import SegMetrics, measure_inference_fps
from .models import build_loss, build_model, param_count_millions
from .visualize import save_triplet_png

__all__ = [
    'BDAPPVSegDataset',
    'SyntheticSegDataset',
    'SegMetrics',
    'build_loss',
    'build_model',
    'list_pairs',
    'measure_inference_fps',
    'param_count_millions',
    'save_triplet_png',
    'split_from_annotations',
    'split_random',
    'take_subset',
]
