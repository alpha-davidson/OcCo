#!/bin/bash

#SBATCH --job-name "OcCo_Serialize"
#SBATCH --output "occo_serialize.out"

source "/opt/conda/bin/activate" "occo-tf"
python render/O16_npy_serialize.py \
    --complete_dir data/O16/occluded/validation_complete.npy \
    --occluded_dir data/O16/occluded/validation_occluded.npy \
    --output_file data/O16/serialized/validation.lmdb