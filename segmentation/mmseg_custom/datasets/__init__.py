# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_wrappers import ConcatDataset
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .nyu_depth_v2 import NYUDepthV2Dataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
import mmseg.datasets.dataset_wrappers as _mmseg_dataset_wrappers

# Ensure mmseg's builder uses the custom ConcatDataset that supports
# Cityscapes concat training/evaluation.
_mmseg_dataset_wrappers.ConcatDataset = ConcatDataset

__all__ = [
    'MapillaryDataset', 'NYUDepthV2Dataset', 'ConcatDataset'
]
