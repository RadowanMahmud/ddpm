#!/bin/bash
# Full evaluation of the 60k snapshots on one A6000.
# Submit from the project folder (the one with runs/ and data/):  sbatch eval_job.slurm
# Another snapshot:  sbatch --export=ALL,ITER=50000 eval_job.slurm
# If it hits the time limit, submit again: finished sampling is skipped.

#SBATCH --job-name=ddpm-eval
#SBATCH --account=mlia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniforge
eval "$(conda shell.bash hook)"
set +u; conda activate ddpm; set -u
export PYTHONNOUSERSITE=1

which python
nvidia-smi
bash eval_checkpoint.sh
