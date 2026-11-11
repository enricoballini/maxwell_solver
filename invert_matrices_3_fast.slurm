#!/bin/bash
#SBATCH --job-name=3_8_cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=output_3.txt

apptainer exec "$HOME/container_gpu.sif" python3 -u "$HOME/work_1/maxwell_solver/invert_matrices.py"
