#!/usr/bin/env bash

python train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py  --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/baseline --cfg-options data.workers_per_gpu=32