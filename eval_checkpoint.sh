#!/usr/bin/env bash
# Full evaluation of one saved EMA snapshot of both models (default: iteration 60000).
# Writes to samples_<N>k/ and results_<N>k/, so the main 100k samples/ and results/ are untouched.
# Usage:  bash eval_checkpoint.sh                 (60k)
#         ITER=50000 bash eval_checkpoint.sh      (another snapshot)
# Re-running is safe: finished sampling is skipped.
set -euo pipefail
cd "$(dirname "$0")"

ITER=${ITER:-60000}
UNCOND_COUNT=${UNCOND_COUNT:-2000}  # must be >= 1539 for metrics_all
COND_COUNT=${COND_COUNT:-500}       # per category, must be >= 472
PYTHON=${PYTHON:-python}
TAG=$((ITER / 1000))k
SAMPLES=samples_$TAG
RESULTS=results_$TAG
printf -v STEP "%07d" "$ITER"

for MODE in unconditional conditional; do
  [ -f "runs/${MODE}_ema_${STEP}.pt" ] || { echo "Missing runs/${MODE}_ema_${STEP}.pt"; exit 1; }
done
[ -f runs/classifier.pt ] || $PYTHON classify.py train data/train data/test runs/classifier.pt
mkdir -p "$SAMPLES" "$RESULTS"

# 1. Sample (a .done marker means finished; an interrupted folder is redone)
sample() {  # mode count grid
  if [ ! -f "$SAMPLES/$1.done" ]; then
    rm -rf "$SAMPLES/$1" "$SAMPLES/$1_grid.png"
    $PYTHON run.py sample "runs/${1}_ema_${STEP}.pt" "$SAMPLES/$1" --count "$2" --grid "$3"
    touch "$SAMPLES/$1.done"
  fi
}
sample unconditional "$UNCOND_COUNT" 50
sample conditional "$COND_COUNT" 10

# 2. FID / KID / IS
$PYTHON evaluate.py --real data/test --train-dir data/train \
  --unconditional "$SAMPLES/unconditional" --conditional "$SAMPLES/conditional" --output "$RESULTS/metrics_test.csv"
$PYTHON evaluate.py --real data/train data/test \
  --unconditional "$SAMPLES/unconditional" --conditional "$SAMPLES/conditional" --output "$RESULTS/metrics_all.csv"

# 3. Classifier: category coverage and controllability
$PYTHON classify.py label runs/classifier.pt data/train "$RESULTS" \
  --unconditional "$SAMPLES/unconditional" --conditional "$SAMPLES/conditional"

# 4. Copying / diversity
$PYTHON nearest.py data/train data/test "$SAMPLES/unconditional" "$RESULTS/nearest_unconditional"
$PYTHON nearest.py data/train data/test "$SAMPLES/conditional" "$RESULTS/nearest_conditional"

# 5. Tables (+ grids copied next to them)
cp "$SAMPLES"/*_grid.png "$RESULTS"/
$PYTHON report.py --results "$RESULTS" --runs runs
echo "Done. See $RESULTS/report_tables.md and $RESULTS/*.png"
