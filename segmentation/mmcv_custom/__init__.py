# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# -*- coding: utf-8 -*-
import torch
import mmcv.parallel._functions as _mmcv_parallel_functions

from .custom_layer_decay_optimizer_constructor import \
    CustomLayerDecayOptimizerConstructor

# Compatibility patch for MMCV 1.x with newer PyTorch where
# torch.nn.parallel._functions._get_stream expects a torch.device.
_orig_get_stream = _mmcv_parallel_functions._get_stream


def _safe_get_stream(device):
    if isinstance(device, int):
        device = torch.device('cuda', device)
    return _orig_get_stream(device)


_mmcv_parallel_functions._get_stream = _safe_get_stream

__all__ = ['CustomLayerDecayOptimizerConstructor',]
