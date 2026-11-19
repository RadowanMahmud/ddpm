"""Train a DDPM (unconditional or class-conditional), or generate PNGs and report grids from a checkpoint."""
import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import time

from PIL import Image, ImageOps
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ddpm import EMA, Diffusion, NoisePredictor
from prepare_sysu import CLASSES, load_photo


def device_for(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled=True):
    """bfloat16 mixed precision on CUDA (A6000 / Ampere and newer); full precision elsewhere."""
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class PadToSquare:
    """Aspect-preserving resize with neutral gray padding (keeps the full frame)."""

    def __init__(self, side):
        self.side = side

    def __call__(self, image):
        return ImageOps.pad(image, (self.side, self.side), Image.Resampling.BILINEAR, color=(128,) * 3)


def image_transform(side):
    return transforms.Compose([PadToSquare(side), transforms.ToTensor()])


def load_folder(folder, side):
    """Load a <category>/<image> folder into memory as [0, 1] tensors and integer labels."""
    data = datasets.ImageFolder(folder, transform=image_transform(side), loader=load_photo)
    if tuple(data.classes) != CLASSES:
        raise ValueError(f"Expected category folders {CLASSES}, found {data.classes}.")
    images = torch.stack([image for image, _ in data])
    return images, torch.tensor(data.targets)


def build_model(config):
    return NoisePredictor(
        width=config["width"], multipliers=tuple(config["multipliers"]), blocks=config["blocks"],
        attention_resolutions=tuple(config["attention"]), dropout=config["dropout"],
        image_size=config["image_size"],
        num_classes=len(CLASSES) if config["mode"] == "conditional" else None,
    )


def generate(model, diffusion, count, side, labels, batch_size, device, amp=True):
    images = []
    for start in range(0, count, batch_size):
        batch_labels = None if labels is None else labels[start:start + batch_size]
        size = min(batch_size, count - start)
        with autocast(device, amp):
            batch = diffusion.sample(model, (size, 3, side, side), batch_labels)
        images.append(((batch + 1) / 2).cpu())
    return torch.cat(images)


def preview(model, diffusion, config, device, path):
    side = config["image_size"]
    if config["mode"] == "conditional":
        labels = torch.arange(len(CLASSES), device=device).repeat_interleave(4)
        images, rows = generate(model, diffusion, len(labels), side, labels, 64, device), 4
    else:
        images, rows = generate(model, diffusion, 20, side, None, 64, device), 10
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=rows)


def train(args, device):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    images, labels = load_folder(args.train_dir, args.image_size)
    images, labels = (images * 2 - 1).to(device), labels.to(device)
    print(f"Loaded {len(images)} training photos at {args.image_size}x{args.image_size}.", flush=True)

    config = {
        "mode": args.mode, "width": args.width, "multipliers": list(args.multipliers),
        "blocks": args.blocks, "attention": list(args.attention), "dropout": args.dropout,
        "image_size": args.image_size, "timesteps": args.timesteps,
    }
    hyper = {
        "iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
        "warmup": args.warmup, "ema_decay": args.ema_decay, "grad_clip": args.grad_clip,
        "flip": not args.no_flip, "seed": args.seed, "train_dir": str(args.train_dir),
    }
    conditional = args.mode == "conditional"
    model = build_model(config).to(device)
    ema = EMA(model, args.ema_decay)
    diffusion = Diffusion(args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iteration = 0

    log_path = args.checkpoint.with_suffix(".loss.csv")
    if args.checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"{args.checkpoint} exists. Pass --resume to continue it.")
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state["config"] != config:
            raise ValueError(f"Checkpoint config {state['config']} differs from {config}.")
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        iteration = state["iteration"]
        print(f"Resumed from iteration {iteration}.", flush=True)
    else:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as log:
            csv.writer(log).writerow(("iteration", "loss", "lr", "seconds"))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    def save(path, full=True):
        state = {"config": config, "hyper": hyper, "iteration": iteration, "ema": ema.model.state_dict()}
        if full:
            state.update(model=model.state_dict(), optimizer=optimizer.state_dict())
        pending = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, pending)
        pending.replace(path)

    start, running, seen = time.time(), 0.0, 0
    while iteration < args.iterations:
        for index in torch.randperm(len(images), device=device).split(args.batch_size):
            if len(index) < args.batch_size or iteration >= args.iterations:
                continue
            batch = images[index]
            if not args.no_flip:
                flip = torch.rand(len(batch), device=device) < 0.5
                batch = torch.where(flip[:, None, None, None], batch.flip(3), batch)
            lr = args.lr * min(1.0, (iteration + 1) / args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            with autocast(device):
                loss = diffusion.loss(model, batch, labels[index] if conditional else None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at iteration {iteration}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)
            iteration += 1
            running, seen = running + loss.item(), seen + 1

            if iteration % args.log_every == 0 or iteration == args.iterations:
                elapsed = time.time() - start
                with log_path.open("a", newline="") as log:
                    csv.writer(log).writerow((iteration, f"{running / seen:.6f}", f"{lr:.2e}", f"{elapsed:.0f}"))
                print(f"iter {iteration:7d} | noise MSE {running / seen:.5f} | lr {lr:.1e} | {elapsed / 60:.1f} min",
                      flush=True)
                running, seen = 0.0, 0
            if iteration % args.save_every == 0 or iteration == args.iterations:
                save(args.checkpoint)
                save(args.checkpoint.with_name(f"{args.checkpoint.stem}_ema_{iteration:07d}.pt"), full=False)
            if args.preview_every and iteration % args.preview_every == 0:
                preview(ema.model, diffusion, config, device,
                        args.checkpoint.with_name(f"{args.checkpoint.stem}_preview") / f"{iteration:07d}.png")


def sample(args, device):
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    conditional = config["mode"] == "conditional"
    if args.category and not conditional:
        raise ValueError("Category selection requires a conditional checkpoint.")
    model = build_model(config).to(device)
    model.load_state_dict(state["model"] if args.raw_weights else state["ema"])
    model.eval()
    diffusion = Diffusion(config["timesteps"]).to(device)
    side = config["image_size"]
    print(f"Sampling from iteration {state['iteration']} ({'raw' if args.raw_weights else 'EMA'} weights).",
          flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    grid = []
    # --count is the total for unconditional generation and per category for conditional generation.
    categories = ([args.category] if args.category else list(CLASSES)) if conditional else [None]
    for category in categories:
        folder = args.output / category if category else args.output
        folder.mkdir(exist_ok=True)
        labels = (torch.full((args.count,), CLASSES.index(category), device=device)
                  if conditional else None)
        images = generate(model, diffusion, args.count, side, labels, args.batch_size, device)
        for index, image in enumerate(images):
            save_image(image, folder / f"{index:05d}.png")
        grid.extend(images[:args.grid])
        print(f"Wrote {len(images)} images to {folder}.", flush=True)
    # Keep the grid outside the sample directory so metrics only see individual images.
    # Conditional grid: one row per category in CLASSES order.
    save_image(grid, args.output.with_name(args.output.name + "_grid.png"), nrow=min(args.grid, 10),
               padding=2, pad_value=1)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("train_dir", type=Path)
    training.add_argument("checkpoint", type=Path)
    training.add_argument("--mode", choices=("unconditional", "conditional"), required=True)
    training.add_argument("--iterations", type=positive_int, default=100_000)
    training.add_argument("--batch-size", type=positive_int, default=64)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--warmup", type=positive_int, default=5000)
    training.add_argument("--ema-decay", type=float, default=0.9999)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--no-flip", action="store_true", help="Disable random horizontal flips.")
    training.add_argument("--image-size", type=positive_int, default=64)
    training.add_argument("--width", type=positive_int, default=128)
    training.add_argument("--multipliers", type=positive_int, nargs="+", default=[1, 2, 2, 2])
    training.add_argument("--blocks", type=positive_int, default=2)
    training.add_argument("--attention", type=positive_int, nargs="*", default=[16],
                          help="Resolutions that get self-attention.")
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--timesteps", type=positive_int, default=1000)
    training.add_argument("--log-every", type=positive_int, default=100)
    training.add_argument("--save-every", type=positive_int, default=10_000)
    training.add_argument("--preview-every", type=int, default=10_000, help="0 disables preview grids.")
    training.add_argument("--resume", action="store_true")
    sampling = commands.add_parser("sample")
    sampling.add_argument("checkpoint", type=Path)
    sampling.add_argument("output", type=Path)
    sampling.add_argument("--category", choices=CLASSES)
    sampling.add_argument("--count", type=positive_int, default=10)
    sampling.add_argument("--batch-size", type=positive_int, default=250)
    sampling.add_argument("--grid", type=positive_int, default=10,
                          help="Images per category (conditional) or in total (unconditional) for the report grid.")
    sampling.add_argument("--raw-weights", action="store_true", help="Use raw instead of EMA weights.")
    for command in (training, sampling):
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    print(f"Device: {device}", flush=True)
    if args.command == "train":
        train(args, device)
    else:
        sample(args, device)


if __name__ == "__main__":
    main()
