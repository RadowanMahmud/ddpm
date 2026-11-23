"""Checkpoint sweep: quality vs memorisation over training.

For every saved EMA snapshot (runs/<mode>_ema_<iteration>.pt) of both models, generate a
fixed number of images with the same seed and measure:
  * FID / KID against the held-out test photos (never seen in training)
  * copy rate: share of samples closer (RMS pixel distance) to a training photo than the
    closest real test photo is, plus the median nearest-training-photo distance
  * classifier view: category entropy (unconditional) or controllability (conditional)

Results are appended to results/sweep.csv one row at a time, so an interrupted run can be
restarted and continues with the missing checkpoints. A plot is written to results/sweep.png.
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from classify import Classifier, load_pngs, predict
from ddpm import Diffusion
from evaluate import by_class, metrics_for
from nearest import nearest
from prepare_sysu import CLASSES
from run import build_model, device_for, generate, load_folder

FIELDS = ("mode", "iteration", "generated_count", "fid_test", "kid_test", "kid_test_std",
          "inception_score", "copy_rate", "gen_nn_median", "test_nn_median",
          "category_entropy", "controllability")


def snapshots(runs, mode, wanted):
    found = {}
    for path in runs.glob(f"{mode}_ema_*.pt"):
        iteration = int(path.stem.rsplit("_", 1)[1])
        if not wanted or iteration in wanted:
            found[iteration] = path
    return dict(sorted(found.items()))


def sample_snapshot(path, folder, count, seed, batch_size, device):
    """Write PNGs for one snapshot. Conditional: count // 5 per category in subfolders."""
    done = folder / ".done"
    if done.exists():
        return
    state = torch.load(path, map_location=device, weights_only=True)
    config = state["config"]
    model = build_model(config).to(device)
    model.load_state_dict(state["ema"])
    model.eval()
    diffusion = Diffusion(config["timesteps"]).to(device)
    side = config["image_size"]
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)  # same noise for every checkpoint -> fair comparison
    if config["mode"] == "conditional":
        per_class = count // len(CLASSES)
        for index, category in enumerate(CLASSES):
            labels = torch.full((per_class,), index, device=device)
            images = generate(model, diffusion, per_class, side, labels, batch_size, device)
            (folder / category).mkdir(exist_ok=True)
            for n, image in enumerate(images):
                save_image(image, folder / category / f"{n:05d}.png")
    else:
        images = generate(model, diffusion, count, side, None, batch_size, device)
        for n, image in enumerate(images):
            save_image(image, folder / f"{n:05d}.png")
    done.touch()


def measure(mode, iteration, folder, real_test, train_images, test_nn, classifier, args, device):
    if mode == "conditional":
        groups = [sorted((folder / c).glob("*.png")) for c in CLASSES]
        paths = [p for g in groups for p in g]
        targets = torch.tensor([i for i, g in enumerate(groups) for _ in g])
    else:
        paths = sorted(folder.glob("*.png"))
    metric_args = SimpleNamespace(image_size=args.image_size, batch_size=100, workers=args.workers, is_splits=10)
    metrics = metrics_for(real_test, paths, metric_args, device, kid_subset_size=min(100, len(real_test), len(paths)))

    images = load_pngs(paths, args.image_size)
    gen_nn, _ = nearest(images, train_images, device)
    row = {
        "mode": mode, "iteration": iteration, "generated_count": len(paths),
        "fid_test": metrics["fid"], "kid_test": metrics["kid"], "kid_test_std": metrics["kid_std"],
        "inception_score": metrics["inception_score"],
        "copy_rate": f"{(gen_nn < test_nn.min()).float().mean():.4f}",
        "gen_nn_median": f"{gen_nn.median():.4f}", "test_nn_median": f"{test_nn.median():.4f}",
        "category_entropy": "", "controllability": "",
    }
    if classifier is not None:
        predictions = predict(classifier, images, device).argmax(1)
        if mode == "conditional":
            row["controllability"] = f"{(predictions == targets).float().mean():.4f}"
        else:
            share = torch.bincount(predictions, minlength=len(CLASSES)).float() / len(predictions)
            share = share[share > 0]
            entropy = -(share * share.log()).sum() / torch.log(torch.tensor(float(len(CLASSES))))
            row["category_entropy"] = f"{abs(entropy.item()):.4f}"
    return row


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for mode in ("unconditional", "conditional"):
        selected = sorted((r for r in rows if r["mode"] == mode), key=lambda r: int(r["iteration"]))
        if not selected:
            continue
        x = [int(r["iteration"]) / 1000 for r in selected]
        axes[0].plot(x, [float(r["fid_test"]) for r in selected], marker="o", label=mode)
        axes[1].plot(x, [float(r["kid_test"]) for r in selected], marker="o", label=mode)
        axes[2].plot(x, [100 * float(r["copy_rate"]) for r in selected], marker="o", label=mode)
    for axis, title in zip(axes, ("FID vs test photos (lower is better)",
                                  "KID vs test photos (lower is better)",
                                  "Copy rate: % closer to a training photo than any test photo")):
        axis.set(xlabel="training iteration (thousands)", title=title)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--samples", type=Path, default=Path("sweep_samples"))
    parser.add_argument("--output", type=Path, default=Path("results/sweep.csv"))
    parser.add_argument("--iterations", type=int, nargs="*", default=[],
                        help="Snapshot iterations to evaluate (default: all from --min-iteration on).")
    parser.add_argument("--min-iteration", type=int, default=40_000,
                        help="Skip earlier snapshots (EMA 0.9999 is still dominated by the random init).")
    parser.add_argument("--count", type=int, default=500, help="Images per checkpoint (conditional: split over 5 categories).")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--classifier", type=Path, default=Path("runs/classifier.pt"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = device_for(args.device)
    print(f"Device: {device}", flush=True)

    train_images, _ = load_folder(args.train_dir, args.image_size)
    test_images, _ = load_folder(args.test_dir, args.image_size)
    test_nn, _ = nearest(test_images, train_images, device)
    real_test = [p for paths in by_class([args.test_dir]).values() for p in paths]
    classifier = None
    if args.classifier.exists():
        state = torch.load(args.classifier, map_location=device, weights_only=True)
        classifier = Classifier(pretrained=False).to(device)
        classifier.load_state_dict(state["model"])

    rows = []
    if args.output.exists():
        with args.output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    finished = {(r["mode"], int(r["iteration"])) for r in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    for mode in ("unconditional", "conditional"):
        found = snapshots(args.runs, mode, set(args.iterations))
        found = {i: p for i, p in found.items() if args.iterations or i >= args.min_iteration}
        if not found:
            print(f"No {mode} snapshots found in {args.runs}.")
        for iteration, path in found.items():
            if (mode, iteration) in finished:
                print(f"== {mode} {iteration} already measured", flush=True)
                continue
            folder = args.samples / mode / f"{iteration:07d}"
            print(f"== {mode} {iteration}: sampling {args.count} images", flush=True)
            sample_snapshot(path, folder, args.count, args.seed, args.batch_size, device)
            row = measure(mode, iteration, folder, real_test, train_images, test_nn, classifier, args, device)
            print(row, flush=True)
            rows.append(row)
            with args.output.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
            plot(rows, args.output.with_suffix(".png"))
    plot(rows, args.output.with_suffix(".png"))
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
