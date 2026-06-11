#!/bin/bash
#SBATCH --job-name=training_internimage_h_mask2former_baseline
#SBATCH --output=slurm/training_internimage_h_mask2former_baseline_%J_%j_%a.out
#SBATCH --error=slurm/training_internimage_h_mask2former_baseline_%J_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --array=0-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de
#SBATCH --gres-flags=enforce-binding


CONFIG="configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py"

seeds=(0 1 2)
preprocessings=("black_white" "color_opponency" "single_color")

seed=${seeds[$SLURM_ARRAY_TASK_ID % ${#seeds[@]}]}
preprocessing=${preprocessings[$SLURM_ARRAY_TASK_ID / ${#seeds[@]}]}

python -W ignore train.py $CONFIG \
    --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing/${preprocessing}/seed_${seed} \
    --seed $seed --deterministic \
    --preprocessing ${preprocessing} \
    --cfg-options data.workers_per_gpu=32



#python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
#    --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/debugging_preprocessing_baseline/seed_0 \
#    --preprocessing black_white \
#    --seed 0 --deterministic --cfg-options data.workers_per_gpu=32 

#python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
#    --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/debugging_preprocessing_baseline/seed_0 \
#    --preprocessing color_opponency \
#    --seed 0 --deterministic --cfg-options data.workers_per_gpu=32

#python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
#    --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/debugging_preprocessing_baseline/seed_0 \
#    --preprocessing single_color \
#    --seed 0 --deterministic --cfg-options data.workers_per_gpu=32