"""Prepare a reproducible ImageFolder train/test split from SYSU-Shape photos."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import random
from shutil import copy2
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

CLASSES = ("airplane", "bicycle", "boat", "car", "motorbike")


def load_photo(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def photo_groups(source):
    groups = defaultdict(list)
    for category in CLASSES:
        folder = source / category / "images"
        paths = sorted(path for path in folder.glob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
        if not paths:
            raise ValueError(f"No SYSU-Shape JPEG photos found in {folder}.")
        for path in paths:
            rgb = load_photo(path)
            key = (rgb.size, hashlib.sha256(rgb.tobytes()).digest())
            groups[key].append((category, path))
    return groups.values()


def prepare(source, output, test_fraction, seed):
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must be between 0 and 1.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory. {output} already exists.")

    by_class = defaultdict(list)
    excluded = []
    for group in photo_groups(source):
        if len({category for category, _ in group}) > 1:
            excluded.extend((category, path, "conflicting_labels") for category, path in group)
            continue
        category, path = group[0]
        by_class[category].append(path)
        excluded.extend((label, duplicate, "duplicate") for label, duplicate in group[1:])

    rng = random.Random(seed)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        for category in CLASSES:
            paths = sorted(by_class[category])
            if len(paths) < 2:
                raise ValueError(f"{category} needs at least two distinct photos for a split.")
            rng.shuffle(paths)
            test_count = min(len(paths) - 1, max(1, round(len(paths) * test_fraction)))
            for split, selected in (("test", paths[:test_count]), ("train", paths[test_count:])):
                folder = staged / split / category
                folder.mkdir(parents=True)
                for path in sorted(selected):
                    destination = folder / path.name
                    copy2(path, destination)
                    rows.append((split, category, path.relative_to(source), destination.relative_to(staged), ""))
            print(f"{category:9} train={len(paths) - test_count:3} test={test_count:3}")

        for category, path, reason in excluded:
            rows.append(("excluded", category, path.relative_to(source), "", reason))
        with (staged / "split_manifest.csv").open("w", newline="") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(("split", "category", "source", "destination", "reason"))
            writer.writerows(rows)
        staged.rename(output)
    print(f"Excluded {len(excluded)} duplicate or conflicting-label entries. See {output / 'split_manifest.csv'}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root of the cloned SYSU-Shape repository")
    parser.add_argument("output", type=Path, help="New directory for the prepared dataset")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.test_fraction, args.seed)


if __name__ == "__main__":
    main()
