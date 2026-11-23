#!/bin/bash
# Checkpoint sweep (quality vs copying over training) on one A6000.
# Submit from the project folder (the one with runs/ and data/):  sbatch sweep_job.slurm
# If it hits the time limit, submit again: finished checkpoints are skipped.

#SBATCH --job-name=ddpm-sweep
#SBATCH --account=mlia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniforge
eval "$(conda shell.bash hook)"
set +u; conda activate ddpm; set -u
export PYTHONNOUSERSITE=1

which python
nvidia-smi
python sweep.py
python report.py
