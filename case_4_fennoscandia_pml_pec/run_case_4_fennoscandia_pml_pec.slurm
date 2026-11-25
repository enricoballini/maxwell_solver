#!/bin/bash
#SBATCH --job-name=run_case_4_fennoscandia_pml_pec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_run_case_4_fennoscandia_pml_pec.txt

apptainer exec --nv "$HOME/container_gpu.sif" bash "$HOME/work_1/maxwell_solver/case_4_fennoscandia_pml_pec/run.sh"
