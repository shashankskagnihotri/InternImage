# Cityscapes + ACDC + DarkZurich Setup Notes

## Thinking / Approach
- Goal: finish dataset wiring for Cityscapes extraTrainVal annotations, add automatic post-training validation on Cityscapes val + ACDC (rain/fog/night/snow) + DarkZurich, support `--test-only` with `--checkpoint-path`, and persist results to CSV in `work_dir`.
- Main risk observed during execution was runtime compatibility (SciPy/libstdc++) and memory constraints under the current Slurm allocation.
- I prioritized making the pipeline robust in code first, then validating execution end-to-end with available resources.

## Steps Taken

### 1) Dataset preparation (extraTrainVal)
- Input zip: `/ceph/sagnihot/datasets/refinement_final_v0.zip`
- Extracted annotations under Cityscapes root:
  - `/ceph/sagnihot/datasets/cityscapes/refinement_final/...`
- Found annotation label space mismatch (raw Cityscapes label IDs, not trainIds).
- Converted raw labels to trainIds into:
  - `/ceph/sagnihot/datasets/cityscapes/refinement_final_trainIds/train_extra/...`
- Validation checks:
  - `train_extra_images`: `19998`
  - `ann_raw`: `19999`
  - `ann_trainIds`: `19999`
- Note: one extra annotation exists without a paired image (training is image-indexed, so this does not break loading).

### 2) Config/data wiring
- Config updated to train on concat dataset:
  - Cityscapes fine train (`leftImg8bit/train` + `gtFine/train`)
  - extraTrainVal trainIds (`leftImg8bit/train_extra` + `refinement_final_trainIds/train_extra`)
- Data roots set to absolute dataset paths.

### 3) Training/evaluation pipeline code
- Added to `segmentation/train.py`:
  - `--test-only`
  - `--checkpoint-path`
  - `--skip-post-validation`
- Automatic post-validation now runs after training (unless skipped) on:
  - `cityscapes_val`
  - `acdc_rain_val`
  - `acdc_fog_val`
  - `acdc_night_val`
  - `acdc_snow_val`
  - `darkzurich_val`
  - plus computed row `acdc_4_mean`
- CSV output implemented in `work_dir/post_validation_results.csv`.
- Logging robustness:
  - ensures text logger hook
  - tensorboard hook is auto-removed if tensorboard backend is unavailable
  - safe config dump fallback when `yapf` verify incompatibility appears.

### 4) Runtime compatibility fixes
- Added conda `libstdc++` preload/re-exec in `train.py` startup to fix SciPy import failure (`GLIBCXX_3.4.30`).
- Added dataloader normalization when `workers_per_gpu=0` to force `persistent_workers=False`.
- Existing repo-level fixes (dataset wrapper / mmcv stream / assigner fallback) were retained and used.

## Validation Results (executed)
- Command mode successfully exercised: `--test-only` with checkpoint and full dataset suite.
- Checkpoint used:
  - `/ceph/sagnihot/projects/InternImage/pretrained_checkpoint/mask2former_internimage_h_896x896_80k_mapillary.pth`
- Output CSV:
  - `/ceph/sagnihot/projects/InternImage/segmentation/work_dirs/testonly_mapillary/post_validation_results.csv`

Rows from CSV:
- `cityscapes_val`: `mIoU=0.8255` (`n=500`)
- `acdc_rain_val`: `mIoU=0.7461` (`n=100`)
- `acdc_fog_val`: `mIoU=0.8365` (`n=100`)
- `acdc_night_val`: `mIoU=0.5877` (`n=106`)
- `acdc_snow_val`: `mIoU=0.7248` (`n=100`)
- `darkzurich_val`: `mIoU=0.5481` (`n=50`)
- `acdc_4_mean`: `mIoU=0.723775`

## Outstanding Problems / Blockers
- I could not complete the requested **500-iteration training** of:
  - `segmentation/configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py`
- Reason: repeated **Slurm cgroup OOM kills** (signal 9) during/after first training iteration under current job memory.
- Current allocation (`scontrol show job 173438`):
  - `ReqTRES mem=10G`
- OOM evidence from `dmesg` shows Python process killed at ~8.4-8.8 GB RSS within `/job_173438/.../task_0` cgroup.

## Recommended Fix to Finish Training
- Re-run on the same GPU type with higher CPU RAM reservation (e.g. at least `24G`, preferably `32G+` for stability with this model/config).
- Keep current code as-is; then run:
  - normal training (500 iters)
  - automatic post-validation (already wired)
  - CSV is automatically written to `work_dir/post_validation_results.csv`.

## Notes
- Tensorboard package is not currently available in env, so tensorboard hook is auto-disabled at runtime. Text logs still work.
- `--test-only` path is verified working end-to-end (including CSV write).
