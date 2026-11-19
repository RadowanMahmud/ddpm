"""Write overall and per-category FID, KID and Inception Score to a CSV.

The real reference set is one or more <category>/<image> folders (e.g. only data/test, or
data/train plus data/test). For each category, the conditional samples are matched to the
number of real photos; the unconditional samples are matched to the total.
Rows: unconditional/overall, conditional/overall, conditional/<each category>, and
real/overall (real test photos vs real train photos, a rough lower bound for FID/KID).
"""
import argparse
import csv
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from prepare_sysu import CLASSES, load_photo
from run import device_for, image_transform, positive_int


class Images(Dataset):
    def __init__(self, paths, side):
        self.paths = paths
        self.transform = image_transform(side)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_photo(self.paths[index]))


def pick(paths, count, rng, description):
    if len(paths) < count:
        raise ValueError(f"{description} has {len(paths)} images. Evaluation needs {count}.")
    return rng.sample(paths, count)


def metrics_for(real, generated, args, device, kid_subset_size):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance

    fid = FrechetInceptionDistance(normalize=True).set_dtype(torch.float64).to(device)
    kid = KernelInceptionDistance(normalize=True, subset_size=kid_subset_size).to(device)
    inception = InceptionScore(normalize=True, splits=args.is_splits).to(device)
    for is_real, paths in ((True, real), (False, generated)):
        loader = DataLoader(Images(paths, args.image_size), batch_size=args.batch_size, num_workers=args.workers)
        for images in loader:
            images = images.to(device)
            fid.update(images, real=is_real)
            kid.update(images, real=is_real)
            if not is_real:
                inception.update(images)
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()
    return {
        "real_count": len(real),
        "generated_count": len(generated),
        "fid": f"{fid.compute().item():.3f}",
        "kid": f"{kid_mean.item():.5f}",
        "kid_std": f"{kid_std.item():.5f}",
        "inception_score": f"{is_mean.item():.3f}",
        "inception_score_std": f"{is_std.item():.3f}",
    }


def by_class(folders):
    result = {category: [] for category in CLASSES}
    for folder in folders:
        data = ImageFolder(folder)
        if tuple(data.classes) != CLASSES:
            raise ValueError(f"Expected category folders {CLASSES} in {folder}, found {data.classes}.")
        for path, target in data.samples:
            result[CLASSES[target]].append(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, nargs="+", required=True, help="Real reference folder(s).")
    parser.add_argument("--unconditional", type=Path, required=True)
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path,
                        help="If given, also add a real-vs-real row: --real photos vs these photos.")
    parser.add_argument(
        "--per-class", type=positive_int,
        help="Use this many real and generated images per class. Default uses all reference photos.",
    )
    parser.add_argument("--image-size", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=100)
    parser.add_argument("--is-splits", type=positive_int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Defaults to cuda when available.")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)

    reference = by_class(args.real)
    real_by_class, conditional_by_class = {}, {}
    for category in CLASSES:
        paths = sorted(reference[category])
        count = args.per_class if args.per_class is not None else len(paths)
        if count < args.is_splits:
            raise ValueError(f"{category} needs at least {args.is_splits} reference photos.")
        real_by_class[category] = pick(paths, count, rng, f"Real {category}")
        conditional_by_class[category] = pick(
            sorted((args.conditional / category).glob("*.png")), count, rng, f"Generated {category}",
        )

    real = [path for paths in real_by_class.values() for path in paths]
    conditional = [path for paths in conditional_by_class.values() for path in paths]
    unconditional = pick(sorted(args.unconditional.glob("*.png")), len(real), rng, "Unconditional samples")
    rng.shuffle(conditional)
    comparisons = [
        ("unconditional", "overall", real, unconditional),
        ("conditional", "overall", real, conditional),
    ]
    comparisons.extend(
        ("conditional", category, real_by_class[category], conditional_by_class[category])
        for category in CLASSES
    )
    if args.train_dir:
        train = [p for paths in by_class([args.train_dir]).values() for p in paths]
        if set(map(str, train)) & set(map(str, real)):
            print("Skipping real-vs-real row: the reference already contains the training photos.")
        else:
            comparisons.append(("real_train", "overall", real, pick(train, min(len(train), len(real)), rng, "Train")))
    kid_subset_size = min(100, min(map(len, real_by_class.values())))
    rows = []
    for mode, category, real_paths, generated_paths in comparisons:
        print(f"Evaluating {mode} / {category} ({len(real_paths)} real, {len(generated_paths)} generated)", flush=True)
        rows.append({
            "mode": mode, "category": category,
            **metrics_for(real_paths, generated_paths, args, device, kid_subset_size),
            "kid_subset_size": kid_subset_size,
        })
        print(rows[-1], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
