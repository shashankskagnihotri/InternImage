# Added Preprocessing Support (segmentation/train.py)

## What I changed
- Added a new CLI argument in `segmentation/train.py`:
  - `--preprocessing {baseline,black_white,color_opponency,single_color}`
- Implemented preprocessing construction for the 3 requested modes using `BlurPreprocessing` with the exact parameter values you requested, and only replacing `path` with the active `cfg.work_dir`.
- Kept `baseline` as default/no-op behavior.
- Added a model-forward hook so preprocessing is applied immediately before model forward (`model(inputs)` equivalent) in:
  - training
  - post-validation / test-only evaluation
- Did **not** modify `segmentation/blur_preprocessing.py`.
- Ensured preprocessing module is not registered as model parameters (so optimizer/checkpoint behavior is unchanged except input preprocessing itself).

## Constructor mapping used
- `--preprocessing color_opponency`
  - `BlurPreprocessing(blur_bool=True, blur_depth=5, single_color=False, color_opponency=True, channels=3, path=<work_dir>, training=False, black_white=False, normalize=False, sparsity_threshold=0.7, sparsity_type='percentage', change_range=(0, 1))`
- `--preprocessing black_white`
  - `BlurPreprocessing(blur_bool=True, blur_depth=5, single_color=False, color_opponency=False, channels=3, path=<work_dir>, training=False, black_white=True, normalize=False, sparsity_threshold=0.7, sparsity_type='percentage', change_range=(0, 1))`
- `--preprocessing single_color`
  - `BlurPreprocessing(blur_bool=True, blur_depth=5, single_color=True, color_opponency=False, channels=3, path=<work_dir>, training=False, black_white=False, normalize=False, sparsity_threshold=0.7, sparsity_type='percentage', change_range=(0, 1))`

## Commands used (1000-iter runs)
Run from `segmentation/` in conda env `internimage`:

```bash
conda activate internimage
cd /ceph/sagnihot/projects/InternImage/segmentation

python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
  --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing_black_white_1000 \
  --preprocessing black_white \
  --cfg-options data.workers_per_gpu=32 runner.max_iters=1000

python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
  --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing_color_opponency_1000 \
  --preprocessing color_opponency \
  --cfg-options data.workers_per_gpu=32 runner.max_iters=1000

python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
  --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing_single_color_1000 \
  --preprocessing single_color \
  --cfg-options data.workers_per_gpu=32 runner.max_iters=1000
```

## What works
- Training works for all 3 preprocessing types through `1000/1000` iterations.
- Automatic post-validation runs successfully for all required datasets:
  - `cityscapes_val`
  - `acdc_rain_val`
  - `acdc_fog_val`
  - `acdc_night_val`
  - `acdc_snow_val`
  - `darkzurich_val`
  - aggregated `acdc_4_mean`
- Artifacts are saved correctly for all 3 runs:
  - checkpoints: `iter_1000.pth`, `latest.pth`
  - metrics CSV: `post_validation_results.csv`
  - preprocessing images: `images_test/` with 8 files (`image_before*` and `image_after*` RGB/r/g/b)

## What does not work / caveats
- No blocking failures observed in the final runs.
- `TensorboardLoggerHook` is auto-removed because TensorBoard backend is unavailable in this env (warning only; training/eval unaffected).

## 1000-iteration results
Numbers below are from each run’s `post_validation_results.csv` and training log `Iter [1000/1000]`.

| preprocessing | iter1000 loss | cityscapes_val mIoU | acdc_rain mIoU | acdc_fog mIoU | acdc_night mIoU | acdc_snow mIoU | acdc_4_mean mIoU | darkzurich_val mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| black_white | 27.4212 | 0.759400 | 0.688800 | 0.720100 | 0.409900 | 0.703600 | 0.630600 | 0.374700 |
| color_opponency | 26.4095 | 0.770900 | 0.694700 | 0.730100 | 0.389700 | 0.699900 | 0.628600 | 0.338600 |
| single_color | 27.6221 | 0.765900 | 0.689400 | 0.715400 | 0.409300 | 0.702200 | 0.629075 | 0.370800 |

## Files touched
- `segmentation/train.py`
- `AddedPreprocessing.md`

## Notes
- I did not install any extra libraries because none were needed for this change.
