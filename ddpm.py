"""DDPM U-Net (Ho et al. 2020, Appendix B), EMA weights, and the DDPM training and sampling rules.

The default network follows the paper's CIFAR-10 setup: base width 128, channel multipliers
(1, 2, 2, 2), two residual blocks per resolution, self-attention at 16 x 16, dropout 0.1.
For class-conditional generation, a learned category embedding is added to the timestep
embedding. Everything else is shared between the two modes.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, dropout):
        super().__init__()
        self.first = nn.Sequential(
            norm(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, out_channels))
        self.second = nn.Sequential(
            norm(out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.second[-1].weight)
        nn.init.zeros_(self.second[-1].bias)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, images, embedding):
        hidden = self.first(images) + self.condition(embedding)[:, :, None, None]
        return self.skip(images) + self.second(hidden)


class Attention(nn.Module):
    """Single-head self-attention over spatial positions, as in the DDPM U-Net."""

    def __init__(self, channels):
        super().__init__()
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, images, embedding=None):
        batch, channels, height, width = images.shape
        qkv = self.qkv(self.norm(images)).reshape(batch, 3, channels, height * width)
        query, key, value = (part.transpose(1, 2)[:, None] for part in qkv.unbind(1))
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden[:, 0].transpose(1, 2).reshape(batch, channels, height, width)
        return images + self.out(hidden)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, images, embedding=None):
        return self.conv(images)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, images, embedding=None):
        return self.conv(F.interpolate(images, scale_factor=2, mode="nearest"))


class Stage(nn.ModuleList):
    def forward(self, images, embedding):
        for layer in self:
            images = layer(images, embedding)
        return images


class NoisePredictor(nn.Module):
    def __init__(
        self, width=128, multipliers=(1, 2, 2, 2), blocks=2, attention_resolutions=(16,),
        dropout=0.1, image_size=64, num_classes=None,
    ):
        super().__init__()
        if width < 32 or width % 32:
            raise ValueError("Width must be a positive multiple of 32.")
        if image_size % 2 ** (len(multipliers) - 1):
            raise ValueError("Image size must be divisible by 2^(levels - 1).")
        embedding_dim = width * 4
        frequencies = torch.exp(-math.log(10_000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequencies", frequencies)
        self.time = nn.Sequential(
            nn.Linear(width, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.category = nn.Embedding(num_classes, embedding_dim) if num_classes else None

        def block(cin, cout, resolution):
            layers = [ResidualBlock(cin, cout, embedding_dim, dropout)]
            if resolution in attention_resolutions:
                layers.append(Attention(cout))
            return Stage(layers)

        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.down = nn.ModuleList()
        skips = [width]
        channels, resolution = width, image_size
        for level, multiplier in enumerate(multipliers):
            for _ in range(blocks):
                self.down.append(block(channels, width * multiplier, resolution))
                channels = width * multiplier
                skips.append(channels)
            if level < len(multipliers) - 1:
                self.down.append(Stage([Downsample(channels)]))
                skips.append(channels)
                resolution //= 2

        self.middle = Stage([
            ResidualBlock(channels, channels, embedding_dim, dropout),
            Attention(channels),
            ResidualBlock(channels, channels, embedding_dim, dropout),
        ])

        self.up = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(multipliers))):
            for _ in range(blocks + 1):
                self.up.append(block(channels + skips.pop(), width * multiplier, resolution))
                channels = width * multiplier
            if level > 0:
                self.up.append(Stage([Upsample(channels)]))
                resolution *= 2

        self.output = nn.Sequential(norm(channels), nn.SiLU(), nn.Conv2d(channels, 3, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, images, timesteps, labels=None):
        angles = timesteps.float()[:, None] * self.frequencies[None, :]
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), dim=1))
        if self.category is not None:
            if labels is None:
                raise ValueError("Conditional generation requires category labels.")
            embedding = embedding + self.category(labels)

        hidden = [self.input(images)]
        for stage in self.down:
            hidden.append(stage(hidden[-1], embedding))
        images = self.middle(hidden[-1], embedding)
        for stage in self.up:
            if isinstance(stage[0], Upsample):
                images = stage(images, embedding)
            else:
                images = stage(torch.cat((images, hidden.pop()), dim=1), embedding)
        return self.output(images)


class EMA:
    """Exponential moving average of model weights (the paper uses decay 0.9999)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(current, 1 - self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)


class Diffusion(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if steps < 2:
            raise ValueError("At least two diffusion steps are required.")
        self.steps = steps
        # Index 0 corresponds to t=1 in the paper. Linear schedule from 1e-4 to 0.02.
        beta = torch.linspace(0.0001, 0.02, steps)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(dim=0)
        previous_alpha_bar = F.pad(alpha_bar[:-1], (1, 0), value=1)
        self.register_buffer("signal_scale", alpha_bar.sqrt())
        self.register_buffer("noise_scale", (1 - alpha_bar).sqrt())
        self.register_buffer("reverse_scale", alpha.rsqrt())
        self.register_buffer("reverse_noise_weight", beta / self.noise_scale)
        self.register_buffer(
            "posterior_std", (beta * (1 - previous_alpha_bar) / (1 - alpha_bar)).sqrt()
        )

    def add_noise(self, clean, timesteps, noise):
        signal = self.signal_scale[timesteps, None, None, None]
        noise_scale = self.noise_scale[timesteps, None, None, None]
        return signal * clean + noise_scale * noise

    def loss(self, model, clean, labels=None):
        """Algorithm 1: simplified noise-prediction objective."""
        timesteps = torch.randint(self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.add_noise(clean, timesteps, noise)
        return F.mse_loss(model(noisy, timesteps, labels).float(), noise)

    @torch.inference_mode()
    def sample(self, model, shape, labels=None):
        """Algorithm 2: ancestral sampling from pure Gaussian noise."""
        device = self.signal_scale.device
        images = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = model(images, timesteps, labels).float()
            images = self.reverse_scale[step] * (
                images - self.reverse_noise_weight[step] * predicted_noise
            )
            if step > 0:
                images = images + self.posterior_std[step] * torch.randn_like(images)
        return images.clamp(-1, 1)
