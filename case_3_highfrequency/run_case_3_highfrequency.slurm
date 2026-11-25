#!/bin/bash
#SBATCH --job-name=run_case_3_highfrequency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_run_case_3_highfrequency.txt

apptainer exec --nv "$HOME/container_gpu.sif" bash "$HOME/work_1/maxwell_solver/case_3_highfrequency/run.sh"
