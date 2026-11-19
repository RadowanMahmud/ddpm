#!/usr/bin/env bash
# Full pipeline for one GPU (tested sizes assume an RTX A6000, 48 GB).
# Usage:  bash run_all.sh            (override any setting, e.g. ITERATIONS=50000 bash run_all.sh)
# Re-running is safe: finished stages are skipped, and an interrupted training run resumes.
set -euo pipefail
cd "$(dirname "$0")"

ITERATIONS=${ITERATIONS:-100000}   # training iterations per model (batch 64)
UNCOND_COUNT=${UNCOND_COUNT:-2000} # unconditional samples (must be >= 1539 for metrics_all)
COND_COUNT=${COND_COUNT:-500}      # conditional samples per category (must be >= 472)
PYTHON=${PYTHON:-python}

# 1. Data -------------------------------------------------------------------------------
if [ ! -d sysu-shape-dataset ]; then
  git clone https://github.com/bearpaw/sysu-shape-dataset.git
  git -C sysu-shape-dataset checkout f2a453474b2711485e2c1bb14eb989bc4133d9e1
fi
[ -d data ] || $PYTHON prepare_sysu.py sysu-shape-dataset data

# 2. Train both DDPMs with identical settings ---------------------------------------------
mkdir -p runs
for MODE in unconditional conditional; do
  CKPT=runs/$MODE.pt
  if [ -f "$CKPT" ] && $PYTHON -c "import torch,sys; s=torch.load('$CKPT',weights_only=True); sys.exit(s['iteration']<$ITERATIONS)"; then
    echo "== $MODE already trained"; continue
  fi
  RESUME=""; [ -f "$CKPT" ] && RESUME="--resume"
  $PYTHON run.py train data/train "$CKPT" --mode $MODE --iterations "$ITERATIONS" $RESUME 2>&1 | tee -a runs/$MODE.log
done

# 3. Sample --------------------------------------------------------------------------------
# A .done marker means sampling finished; a folder without it was interrupted and is redone.
sample() {  # mode count grid
  if [ ! -f samples/$1.done ]; then
    rm -rf samples/$1 samples/$1_grid.png
    $PYTHON run.py sample runs/$1.pt samples/$1 --count "$2" --grid "$3"
    touch samples/$1.done
  fi
}
sample unconditional "$UNCOND_COUNT" 50
sample conditional "$COND_COUNT" 10

# 4. Metrics: vs held-out test photos, and vs all real photos -----------------------------
$PYTHON evaluate.py --real data/test --train-dir data/train \
  --unconditional samples/unconditional --conditional samples/conditional --output results/metrics_test.csv
$PYTHON evaluate.py --real data/train data/test \
  --unconditional samples/unconditional --conditional samples/conditional --output results/metrics_all.csv

# 5. Classifier analysis (coverage of unconditional, controllability of conditional) -------
[ -f runs/classifier.pt ] || $PYTHON classify.py train data/train data/test runs/classifier.pt
$PYTHON classify.py label runs/classifier.pt data/train results \
  --unconditional samples/unconditional --conditional samples/conditional

# 6. Memorisation / diversity -------------------------------------------------------------
$PYTHON nearest.py data/train data/test samples/unconditional results/nearest_unconditional
$PYTHON nearest.py data/train data/test samples/conditional results/nearest_conditional

# 7. Tables + loss curves -----------------------------------------------------------------
$PYTHON report.py
echo "Done. See results/report_tables.md, samples/*_grid.png, results/nearest_*.png, runs/*_preview/."
