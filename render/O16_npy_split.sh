#!/bin/bash

#SBATCH --job-name "OcCo_Split"
#SBATCH --output "occo_split.out"

source "/opt/conda/bin/activate" "occo-data"
python render/O16_npy_split.py \
    --input_path data/O16/split/train_validation.npy \
    --train_path data/O16/split/train.npy \
    --test_path data/O16/split/validation.npy \
    --split 0.8