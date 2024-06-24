#!/bin/bash

#SBATCH --job-name "OcCo_Occlude"
#SBATCH --output "occo_occlude.out"

source "/opt/conda/bin/activate" "occo-data"
python render/O16_npy_occlude.py