"""Memorisation and diversity check in pixel space.

Memorisation: for each generated image, find its nearest training photo (RMS pixel distance).
The held-out test photos give a baseline: a real but unseen photo is this far from the training
set. Generated images that are much closer than any test photo are likely copies.

Diversity: mean pairwise distance among generated images vs among real photos of the same
group. A much smaller value for generated images suggests low diversity / partial mode collapse.

Writes <prefix>_stats.csv plus two figures: random samples and the closest samples, each shown
above its nearest training photo.
"""
import argparse
import csv
from pathlib import Path
import random

import torch
from torchvision.utils import save_image

from classify import load_pngs
from prepare_sysu import CLASSES
from run import device_for, load_folder


def nearest(queries, references, device, chunk=512):
    references = references.flatten(1).to(device)
    distances, indices = [], []
    for start in range(0, len(queries), chunk):
        batch = queries[start:start + chunk].flatten(1).to(device)
        d = torch.cdist(batch, references)
        value, index = d.min(1)
        distances.append(value.cpu())
        indices.append(index.cpu())
    scale = references.shape[1] ** 0.5
    return torch.cat(distances) / scale, torch.cat(indices)


def mean_pairwise(images, device, limit=1000, seed=0):
    if len(images) > limit:
        images = images[torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))[:limit]]
    flat = images.flatten(1).to(device)
    d = torch.cdist(flat, flat) / flat.shape[1] ** 0.5
    n = len(flat)
    return (d.sum() / (n * (n - 1))).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("test_dir", type=Path)
    parser.add_argument("samples", type=Path, help="Unconditional folder, or conditional folder with category subfolders.")
    parser.add_argument("prefix", type=Path, help="Output prefix, e.g. results/nearest_unconditional")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--show", type=int, default=10, help="Images per figure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = device_for(args.device)

    train_images, train_labels = load_folder(args.train_dir, args.image_size)
    test_images, test_labels = load_folder(args.test_dir, args.image_size)
    conditional = all((args.samples / c).is_dir() for c in CLASSES)
    if conditional:
        groups = {c: sorted((args.samples / c).glob("*.png")) for c in CLASSES}
    else:
        groups = {"all": sorted(args.samples.glob("*.png"))}
    paths = [p for group in groups.values() for p in group]
    generated = load_pngs(paths, args.image_size)
    group_of = [name for name, group in groups.items() for _ in group]

    gen_distance, gen_index = nearest(generated, train_images, device)
    test_distance, _ = nearest(test_images, train_images, device)
    threshold = test_distance.min().item()

    rows = []

    def row(name, gen_mask, real_mask_train, real_mask_test):
        g = gen_distance[gen_mask]
        t = test_distance[real_mask_test]
        rows.append({
            "group": name,
            "generated_count": int(gen_mask.sum()),
            "gen_nn_mean": f"{g.mean():.4f}", "gen_nn_median": f"{g.median():.4f}", "gen_nn_min": f"{g.min():.4f}",
            "test_nn_mean": f"{t.mean():.4f}", "test_nn_median": f"{t.median():.4f}", "test_nn_min": f"{t.min():.4f}",
            "share_closer_than_any_test": f"{(g < threshold).float().mean():.4f}",
            "gen_pairwise": f"{mean_pairwise(generated[gen_mask], device):.4f}",
            "train_pairwise": f"{mean_pairwise(train_images[real_mask_train], device):.4f}",
        })

    everything = torch.ones(len(generated), dtype=torch.bool)
    row("overall", everything, torch.ones(len(train_images), dtype=torch.bool),
        torch.ones(len(test_images), dtype=torch.bool))
    if conditional:
        names = torch.tensor([CLASSES.index(g) for g in group_of])
        for index, category in enumerate(CLASSES):
            row(category, names == index, train_labels == index, test_labels == index)

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.prefix.with_name(args.prefix.name + "_stats.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = random.Random(args.seed)
    shown = min(args.show, len(generated))
    random_pick = rng.sample(range(len(generated)), shown)
    closest_pick = gen_distance.argsort()[:shown].tolist()
    for name, pick in (("random", random_pick), ("closest", closest_pick)):
        figure = torch.cat((generated[pick], train_images[gen_index[pick]]))
        save_image(figure, args.prefix.with_name(f"{args.prefix.name}_{name}.png"), nrow=shown, padding=2, pad_value=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
