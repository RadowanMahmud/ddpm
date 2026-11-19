#!/bin/bash
# Full DDPM pipeline on one A6000 on UVA Rivanna/Afton.
# Set --account and CONDA_ENV below, then:  sbatch ddpm_job.slurm
# If the job hits the time limit, submit it again: it resumes where it stopped.

#SBATCH --job-name=ddpm-sysu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=ddpm_%j.out
#SBATCH --error=ddpm_%j.out

CONDA_ENV=ddpm   # name of your conda environment

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniforge
set +u; source activate "$CONDA_ENV"; set -u

nvidia-smi
bash run_all.sh
