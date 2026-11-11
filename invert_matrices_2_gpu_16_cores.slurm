#!/bin/bash
#SBATCH --job-name=2_gpu_16_cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding
#SBATCH --time=00:30:00
#SBATCH --output=output_2.txt

apptainer exec --nv "$HOME/container_gpu.sif" python3 -u "$HOME/work_1/maxwell_solver/invert_matrices.py"
