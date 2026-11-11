#!/bin/bash
#SBATCH --job-name=1_gpu_8_cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=output_1.txt

apptainer exec --nv "$HOME/container_gpu.sif" python3 -u "$HOME/work_1/maxwell_solver/invert_matrices.py"
