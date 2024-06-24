#!/bin/bash

#SBATCH --job-name "OcCo_Split"
#SBATCH --output "occo_split.out"

source "/opt/conda/bin/activate" "occo-data"
python render/O16_npy_split.py