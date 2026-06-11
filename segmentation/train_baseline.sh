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
#SBATCH --array=1-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de
#SBATCH --gres-flags=enforce-binding


CONFIG="configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py"
GPUS=1
PORT=${PORT:-29300}

# python -W ignore -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \ # --launcher pytorch \
python -W ignore train.py $CONFIG \
    --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/baseline/seed_${SLURM_ARRAY_TASK_ID} \
    --seed $SLURM_ARRAY_TASK_ID --deterministic \
    --cfg-options data.workers_per_gpu=32


