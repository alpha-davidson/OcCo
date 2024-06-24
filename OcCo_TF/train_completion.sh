#!/bin/bash

#SBATCH --job-name "OcCo_Test"
#SBATCH --output "occo_test.out"
#SBATCH --gpus 1

source "/opt/conda/bin/activate" "occo-tf"
python OcCo_TF/train_completion.py \
    --gpu 0 \
    --lmdb_train data/O16/serialized/train.lmdb \
    --lmdb_valid data/O16/serialized/validation.lmdb \
    --log_dir OcCo_TF/log/ \
    --batch_size 16 \
    --lr_decay \
    --epoch 10 \
    --steps_per_print 10 \
    --steps_per_visu 50 \
    --num_input_points 700 \
    --dataset shapenet8