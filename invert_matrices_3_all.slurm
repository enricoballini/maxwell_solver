#!/bin/bash
#SBATCH --job-name=invert_all24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=output_3.txt

apptainer exec "$HOME/container_gpu.sif" python3 -u invert_matrices.py
