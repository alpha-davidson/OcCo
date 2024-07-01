#!/bin/bash

#SBATCH --job-name "OcCo_Train"
#SBATCH --output "occo_train.out"
#SBATCH --gpus 1

source "/opt/conda/bin/activate" "occo-tf"
python OcCo_TF/train_completion.py \
    --gpu 0 \
    --lmdb_train data/O16/serialized/train.lmdb \
    --lmdb_valid data/O16/serialized/validation.lmdb \
    --log_dir OcCo_TF/log/ \
    --batch_size 16 \
    --num_gt_points 500 \
    --epoch 30 \
    --visu_freq 1 \
    --num_input_points 350 \
    --dataset shapenet8