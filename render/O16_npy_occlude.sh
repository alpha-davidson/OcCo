#!/bin/bash

#SBATCH --job-name "OcCo_Occlude"
#SBATCH --output "occo_occlude.out"

source "/opt/conda/bin/activate" "occo-data"
python render/O16_npy_occlude.py \
    --input_path data/O16/split/train.npy \
    --complete_path data/O16/occluded/train_complete.npy \
    --occluded_path data/O16/occluded/train_occluded.npy

python render/O16_npy_occlude.py \
    --input_path data/O16/split/validation.npy \
    --complete_path data/O16/occluded/validation_complete.npy \
    --occluded_path data/O16/occluded/validation_occluded.npy