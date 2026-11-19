# DDPM: unconditional vs class-conditional generation on SYSU-Shape

This project trains two Denoising Diffusion Probabilistic Models ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) on five SYSU-Shape object categories:

- **Experiment A (unconditional):** trained on all five categories with no labels, and sampled from pure noise.
- **Experiment B (class-conditional):** same data and network, plus the category label. You can ask for an airplane, bicycle, boat, car or motorbike.

Both models share the same U-Net, linear noise schedule (β from 1e-4 to 0.02, T = 1000), noise-prediction loss (Algorithm 1) and ancestral sampler (Algorithm 2). The only difference is that the conditional model adds a learned category embedding to the timestep embedding.

## What follows the paper

| Setting | Value | Source |
|---|---|---|
| U-Net | width 128, multipliers (1,2,2,2), 2 residual blocks per level, self-attention at 16×16, GroupNorm, dropout 0.1 (~36M parameters) | Paper, Appendix B (CIFAR-10 setup) |
| Optimiser | Adam, lr 2e-4, 5,000-step warmup, gradient clipping at 1.0 | Paper |
| EMA of weights | decay 0.9999. Sampling uses the EMA weights. | Paper |
| Augmentation | random horizontal flip | Paper (CIFAR-10) |
| Batch size | 64 | Our choice (the paper used 128). The dataset is small. |
| Resolution | 64×64, aspect-preserving resize with gray padding | Our choice |
| Mixed precision | bfloat16 on CUDA | Speed only |

## Files

| File | Purpose |
|---|---|
| `prepare_sysu.py` | Builds a stratified 80/20 train/test split from `<category>/images/` |
| `ddpm.py` | U-Net, EMA, and the diffusion process |
| `run.py` | `train` (logs, checkpoints, resume, preview grids) and `sample` (PNGs + report grid) |
| `evaluate.py` | FID, KID and IS, overall and per category, plus a real-vs-real baseline |
| `classify.py` | Independent ResNet-18 classifier. Measures category coverage of unconditional samples and controllability of conditional samples. |
| `nearest.py` | Nearest training image for each sample (memorization check) and pairwise distance (diversity) |
| `report.py` | Collects every result into `results/report_tables.md` and plots `results/loss_curves.png` |
| `run_all.sh` | Runs the whole pipeline in order. Safe to re-run: finished stages are skipped and training resumes. |

## Data

Run `prepare_sysu.py` on a clone of [SYSU-Shape](https://github.com/bearpaw/sysu-shape-dataset) (pinned commit `f2a4534`). It uses the original photos in the five `<category>/images/` folders. The folder name is the label.

The release has 1,541 photos. One identical photo appears under both car and motorbike, so both copies are excluded. The split is stratified by category, 20% test, seed 42. `data/split_manifest.csv` records every file.

| Category | Source files | Train | Test |
|---|---:|---:|---:|
| Airplane | 472 | 378 | 94 |
| Bicycle | 224 | 179 | 45 |
| Boat | 196 | 157 | 39 |
| Car | 302 | 241 | 60 |
| Motorbike | 347 | 277 | 69 |
| **Total** | **1,541** | **1,232** | **307** |

The categories are imbalanced (airplanes are 31% of the training photos). Keep this in mind when judging whether the unconditional model "favors" a category: `classify.py` compares the generated shares to the training shares.

Dataset citation: Liang Lin, Xiaolong Wang, Wei Yang, and JianHuang Lai, [*Discriminatively Trained And-Or Graph Models for Object Shape Detection*](https://doi.org/10.1109/TPAMI.2014.2359888), IEEE TPAMI.

## Running on the cloud GPU (RTX A6000)

```bash
pip install -r requirements.txt
bash run_all.sh                      # everything, default 100k iterations per model
ITERATIONS=50000 bash run_all.sh     # shorter run
```

Rough time on an A6000: each model trains at about 8–12 iterations/s, so 100k iterations take about 2.5–3.5 h per model. Sampling 2,000 + 2,500 images with 1,000 steps takes about an hour. Evaluation and analysis take a few minutes. Run inside `tmux` or `nohup` so a disconnect does not kill the job. If it stops, run `bash run_all.sh` again and it resumes.

Check training progress while it runs:
- `runs/<mode>.loss.csv` and `runs/<mode>.log`: the loss
- `runs/<mode>_preview/<iteration>.png`: EMA samples every 10k iterations. Use these to see when quality stops improving.
- `runs/<mode>_ema_<iteration>.pt`: an EMA snapshot every 10k iterations. To sample an earlier one, pass it to `run.py sample`.

The dataset is small (1,232 training photos), so long training can start memorizing. `nearest.py` checks for this. If the closest-match figure shows copies of training photos, sample an earlier snapshot.

### On UVA Rivanna/Afton (Slurm)

First create a conda env with `requirements.txt` installed. Then:

```bash
cd /scratch/$USER/DDPM_Starter        # scratch is faster and larger than home
nano ddpm_job.slurm                   # set --account (see `allocations`) and CONDA_ENV
sbatch ddpm_job.slurm
squeue -u $USER                       # job status
tail -f ddpm_<jobid>.out              # live log
```

The job asks for 1 A6000, 8 CPUs, 64 GB RAM and 24 hours. If it runs out of time, run `sbatch ddpm_job.slurm` again and it continues where it stopped. To change settings, pass them at submit time: `sbatch --export=ALL,ITERATIONS=50000 ddpm_job.slurm`.

### Individual commands

```bash
python prepare_sysu.py sysu-shape-dataset data
python run.py train data/train runs/unconditional.pt --mode unconditional
python run.py train data/train runs/conditional.pt   --mode conditional
python run.py sample runs/unconditional.pt samples/unconditional --count 2000 --grid 50
python run.py sample runs/conditional.pt   samples/conditional   --count 500  --grid 10   # count is per category
python evaluate.py --real data/test --train-dir data/train \
    --unconditional samples/unconditional --conditional samples/conditional --output results/metrics_test.csv
python evaluate.py --real data/train data/test \
    --unconditional samples/unconditional --conditional samples/conditional --output results/metrics_all.csv
python classify.py train data/train data/test runs/classifier.pt
python classify.py label runs/classifier.pt data/train results --unconditional samples/unconditional --conditional samples/conditional
python nearest.py data/train data/test samples/unconditional results/nearest_unconditional
python nearest.py data/train data/test samples/conditional   results/nearest_conditional
python report.py
```

Add `--resume` to a `train` command to continue an interrupted run. Add `--category boat` to `sample` to generate only one category.

## Outputs and how they map to the assignment

| Requirement | Where |
|---|---|
| FID / KID / IS table, both experiments | `results/metrics_test.csv`, `results/metrics_all.csv` → `report_tables.md` |
| Per-category and overall results for the conditional model | Same files, `conditional/<category>` rows |
| ≥10 unconditional images | `samples/unconditional_grid.png` (50 images) |
| ≥10 conditional images, all 5 categories | `samples/conditional_grid.png` (10 per category, one row each, in the order above) |
| Is unconditional generation biased toward some categories, or collapsing? | `results/classifier_unconditional.csv`: generated share vs training share, histogram entropy |
| Does conditioning improve controllability? (optional classifier task) | `results/classifier_conditional.csv`: confusion matrix and accuracy for the requested category |
| Diversity and memorization | `results/nearest_*_stats.csv`, `results/nearest_*_{random,closest}.png` |
| Training behavior | `results/loss_curves.png`, `runs/*_preview/` |

## Notes on the metrics

- **Two reference sets.** `metrics_test` compares samples to the 307 held-out photos, which the model never saw. `metrics_all` compares them to all 1,539 real photos, which gives more stable numbers. Report both, and say which is which.
- **Sample counts.** The generated set is matched in size to the real set: overall, and per category for the conditional model. FID is biased at these small sizes, so compare the two models only at equal counts. Don't compare these numbers with values from papers.
- **KID** is unbiased for small sets, so it is the most reliable distribution metric here. Each category has only 39–94 test photos, so lean on KID when discussing per-category results.
- **Real-vs-real row** (`real_train`): FID/KID between the real test and train photos. This is roughly the best score you could expect with this many images, and it is a useful scale for the table.
- **Inception Score** uses ImageNet classes, and none of our categories map cleanly to them. Per-category IS mostly measures confidence, not variety within the category.
- The Inception weights are downloaded on first use. The classifier uses ImageNet-pretrained ResNet-18 weights from torchvision. Pass `--no-pretrained` if downloads are blocked.
