#!/bin/bash
#SBATCH --job-name=testing_internimage_h_mask2former_all
#SBATCH --output=slurm/testing_internimage_h_mask2former_all_%J_%j_%a.out
#SBATCH --error=slurm/testing_internimage_h_mask2former_all_%J_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=7:00:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --array=1-11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de
#SBATCH --gres-flags=enforce-binding


CONFIG="configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py"

seeds=(0 1 2)
preprocessings=("baseline" "black_white" "color_opponency" "single_color")

seed=${seeds[$SLURM_ARRAY_TASK_ID / ${#preprocessings[@]}]}
preprocessing=${preprocessings[$SLURM_ARRAY_TASK_ID % ${#preprocessings[@]}]}

sparsities=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

last_checkpoint_name="latest.pth"

if [[ "$preprocessing" == "baseline" ]]; then
    if [[ $seed -eq 0 ]]; then
        last_checkpoint_name="best_mIoU_iter_74000.pth"
    elif [[ $seed -eq 1 ]]; then
        last_checkpoint_name="best_mIoU_iter_70000.pth"
    elif [[ $seed -eq 2 ]]; then
        last_checkpoint_name="best_mIoU_iter_50000.pth"
    fi
elif [[ "$preprocessing" == "black_white" ]]; then
    if [[ $seed -eq 0 ]]; then
        last_checkpoint_name="best_mIoU_iter_78000.pth"
    elif [[ $seed -eq 1 ]]; then
        last_checkpoint_name="best_mIoU_iter_78000.pth"
    elif [[ $seed -eq 2 ]]; then
        last_checkpoint_name="best_mIoU_iter_62000.pth"
    fi
elif [[ "$preprocessing" == "color_opponency" ]]; then
    if [[ $seed -eq 0 ]]; then
        last_checkpoint_name="best_mIoU_iter_52000.pth"
    elif [[ $seed -eq 1 ]]; then
        last_checkpoint_name="best_mIoU_iter_76000.pth"
    elif [[ $seed -eq 2 ]]; then
        last_checkpoint_name="best_mIoU_iter_76000.pth" 
    fi
elif [[ "$preprocessing" == "single_color" ]]; then
    if [[ $seed -eq 0 ]]; then
        last_checkpoint_name="best_mIoU_iter_68000.pth"
    elif [[ $seed -eq 1 ]]; then
        last_checkpoint_name="best_mIoU_iter_78000.pth"
    elif [[ $seed -eq 2 ]]; then
        last_checkpoint_name="best_mIoU_iter_78000.pth"
    fi
fi

for sparsity in "${sparsities[@]}"; do
    python -W ignore train.py $CONFIG \
        --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/testing_preprocessing_with_sparsity/${preprocessing}/seed_${seed}/sparsity_${sparsity} \
        --seed $seed --deterministic \
        --preprocessing ${preprocessing} \
        --sparsity ${sparsity} \
        --test-only \
        --checkpoint-path /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing/${preprocessing}/seed_${seed}/${last_checkpoint_name} \
        --cfg-options data.workers_per_gpu=32
        
done



#python -W ignore train.py configs/cityscapes/mask2former_internimage_h_1024x1024_80k_mapillary2cityscapes.py \
#       --work-dir /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/debugging_testing_preprocessing_with_sparsity/baseline/seed_0/sparsity_0.0 \
#       --seed 0 --deterministic \
#       --preprocessing baseline \
#       --sparsity 0.0 \
#       --test-only \
#       --checkpoint-path /ceph/sagnihot/projects/InternImage/segmentation/work_dirs/preprocessing/baseline/seed_0/best_mIoU_iter_74000.pth \
#       --cfg-options data.workers_per_gpu=32


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