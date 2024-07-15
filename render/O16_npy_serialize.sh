#!/bin/bash

#SBATCH --job-name "OcCo_Serialize"
#SBATCH --output "occo_serialize.out"

source "/opt/conda/bin/activate" "occo-data"
python render/O16_npy_serialize.py \
    --complete_dir data/O16/occluded/train_complete.npy \
    --occluded_dir data/O16/occluded/train_occluded.npy \
    --output_file data/O16/serialized/train.lmdb