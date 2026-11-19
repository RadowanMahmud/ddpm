"""Independent category classifier for analysing generated images.

train: fine-tune an ImageNet ResNet-18 on the real training photos and report held-out accuracy.
label: classify generated images.
  * Unconditional samples -> how often each category appears (coverage / mode collapse / bias).
  * Conditional samples   -> how often the requested category is recognised (controllability).
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from prepare_sysu import CLASSES, load_photo
from run import device_for, image_transform, load_folder

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Classifier(nn.Module):
    """Takes [0, 1] images at the generation resolution, upsamples to 224 for the ImageNet backbone."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(CLASSES))
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, images):
        images = F.interpolate(images, size=224, mode="bilinear", align_corners=False)
        return self.backbone((images - self.mean) / self.std)


def augment(images):
    flip = torch.rand(len(images), device=images.device) < 0.5
    images = torch.where(flip[:, None, None, None], images.flip(3), images)
    side = images.shape[-1]
    pad = side // 8
    padded = F.pad(images, (pad,) * 4, mode="replicate")
    dx, dy = torch.randint(0, 2 * pad + 1, (2,)).tolist()
    return padded[:, :, dy:dy + side, dx:dx + side]


@torch.no_grad()
def predict(model, images, device, batch_size=256):
    model.eval()
    probabilities = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        probabilities.append(model(batch).softmax(dim=1).cpu())
    return torch.cat(probabilities)


def confusion(targets, predictions):
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    return matrix


def write_confusion(path, matrix):
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("intended \\ predicted", *CLASSES, "accuracy"))
        for index, category in enumerate(CLASSES):
            row = matrix[index]
            writer.writerow((category, *row.tolist(), f"{row[index].item() / max(1, row.sum().item()):.4f}"))


def train(args, device):
    torch.manual_seed(args.seed)
    train_images, train_labels = load_folder(args.train_dir, args.image_size)
    test_images, test_labels = load_folder(args.test_dir, args.image_size)
    model = Classifier(pretrained=not args.no_pretrained).to(device)
    # Inverse-frequency class weights so the classifier does not inherit the training-set
    # category imbalance (which would bias the unconditional coverage analysis).
    counts = torch.bincount(train_labels, minlength=len(CLASSES)).float()
    weights = (counts.sum() / (len(CLASSES) * counts)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train_images) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps)
    images, labels = train_images.to(device), train_labels.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for index in torch.randperm(len(images), device=device).split(args.batch_size):
            loss = F.cross_entropy(model(augment(images[index])), labels[index], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(index)
        accuracy = (predict(model, test_images, device).argmax(1) == test_labels).float().mean().item()
        print(f"epoch {epoch:02d} | loss {total / len(images):.4f} | held-out accuracy {accuracy:.4f}", flush=True)

    predictions = predict(model, test_images, device).argmax(1)
    matrix = confusion(test_labels, predictions)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "image_size": args.image_size}, args.checkpoint)
    write_confusion(args.checkpoint.with_suffix(".test_confusion.csv"), matrix)
    summary = {
        "test_accuracy": (predictions == test_labels).float().mean().item(),
        "test_count": len(test_labels),
        "per_class_accuracy": {c: (matrix[i, i] / matrix[i].sum()).item() for i, c in enumerate(CLASSES)},
    }
    args.checkpoint.with_suffix(".test.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_pngs(paths, side):
    transform = image_transform(side)
    return torch.stack([transform(load_photo(path)) for path in paths])


def label(args, device):
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = Classifier(pretrained=False).to(device)
    model.load_state_dict(state["model"])
    side = state["image_size"]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}

    train_counts = torch.tensor([len(list((args.train_dir / c).iterdir())) for c in CLASSES]).float()
    train_share = train_counts / train_counts.sum()

    if args.unconditional:
        paths = sorted(args.unconditional.glob("*.png"))
        probabilities = predict(model, load_pngs(paths, side), device)
        confidence, predictions = probabilities.max(1)
        counts = torch.bincount(predictions, minlength=len(CLASSES))
        share = counts.float() / len(paths)
        with (args.output / "classifier_unconditional.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("category", "generated_count", "generated_share", "training_share",
                             "mean_confidence", "confident_share"))
            for index, category in enumerate(CLASSES):
                mask = predictions == index
                writer.writerow((
                    category, counts[index].item(), f"{share[index]:.4f}", f"{train_share[index]:.4f}",
                    f"{confidence[mask].mean().item() if mask.any() else 0:.4f}",
                    f"{(mask & (confidence >= args.threshold)).float().sum().item() / len(paths):.4f}",
                ))
        # Normalised entropy of the predicted-category histogram: 1 = perfectly even, 0 = one category.
        entropy = -(share[share > 0] * share[share > 0].log()).sum() / torch.log(torch.tensor(len(CLASSES)))
        summary["unconditional"] = {
            "count": len(paths),
            "category_share": dict(zip(CLASSES, share.tolist())),
            "training_share": dict(zip(CLASSES, train_share.tolist())),
            "normalised_entropy": entropy.item(),
            "total_variation_to_training": 0.5 * (share - train_share).abs().sum().item(),
            "mean_confidence": confidence.mean().item(),
            "low_confidence_share": (confidence < args.threshold).float().mean().item(),
        }

    if args.conditional:
        targets, images = [], []
        for index, category in enumerate(CLASSES):
            paths = sorted((args.conditional / category).glob("*.png"))
            images.append(load_pngs(paths, side))
            targets.extend([index] * len(paths))
        targets = torch.tensor(targets)
        probabilities = predict(model, torch.cat(images), device)
        confidence, predictions = probabilities.max(1)
        matrix = confusion(targets, predictions)
        write_confusion(args.output / "classifier_conditional.csv", matrix)
        intended = probabilities[torch.arange(len(targets)), targets]
        summary["conditional"] = {
            "count": len(targets),
            "accuracy": (predictions == targets).float().mean().item(),
            "per_class_accuracy": {c: (matrix[i, i] / matrix[i].sum()).item() for i, c in enumerate(CLASSES)},
            "mean_intended_probability": intended.mean().item(),
            "mean_confidence": confidence.mean().item(),
            "low_confidence_share": (confidence < args.threshold).float().mean().item(),
        }

    test_summary = args.checkpoint.with_suffix(".test.json")
    if test_summary.exists():
        summary["classifier_on_real_test"] = json.loads(test_summary.read_text())
    (args.output / "classifier_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("train_dir", type=Path)
    training.add_argument("test_dir", type=Path)
    training.add_argument("checkpoint", type=Path)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=5e-4)
    training.add_argument("--image-size", type=int, default=64)
    training.add_argument("--no-pretrained", action="store_true")
    training.add_argument("--seed", type=int, default=42)
    labelling = commands.add_parser("label")
    labelling.add_argument("checkpoint", type=Path)
    labelling.add_argument("train_dir", type=Path, help="Used for the training category shares.")
    labelling.add_argument("output", type=Path, help="Directory for the CSV and JSON results.")
    labelling.add_argument("--unconditional", type=Path)
    labelling.add_argument("--conditional", type=Path)
    labelling.add_argument("--threshold", type=float, default=0.5,
                           help="Confidence below which an image counts as not clearly recognisable.")
    for command in (training, labelling):
        command.add_argument("--device")
    args = parser.parse_args()
    device = device_for(args.device)
    train(args, device) if args.command == "train" else label(args, device)


if __name__ == "__main__":
    main()
