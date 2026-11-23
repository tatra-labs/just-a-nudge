"""Prompts and MNIST-composed training images.

A prompt names two digits and a stroke weight for each, e.g. "3 bold, 7 thin":
(left digit, left weight, right digit, right weight) with weight 0 = thin, 1 = regular, 2 = bold.
Its image is two MNIST digits of those classes side by side (28 x 56, scaled to [-1, 1]), each drawn
from its weight's tercile of *that class's* ink (so a bold 1 is bold for a 1). There are
10 x 3 x 10 x 3 = 900 prompts: the base model is pretrained on all of them, RL and steering-vector
fitting use 750 training prompts, and every reported number is on the 150 held-out ones.
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ALL_PROMPTS = [(a, sa, b, sb) for a in range(10) for sa in range(3) for b in range(10) for sb in range(3)]
WEIGHT_NAMES = ("thin", "regular", "bold")


def prompt_split(seed: int = 0, n_test: int = 150):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ALL_PROMPTS), generator=g).tolist()
    test = sorted(ALL_PROMPTS[i] for i in perm[:n_test])
    train = sorted(ALL_PROMPTS[i] for i in perm[n_test:])
    return train, test


def prompt_text(p) -> str:
    return f"{p[0]} {WEIGHT_NAMES[p[1]]}, {p[2]} {WEIGHT_NAMES[p[3]]}"


def load_mnist(train: bool = True):
    import torchvision

    ds = torchvision.datasets.MNIST(str(DATA_DIR), train=train, download=True)
    x = ds.data.float().div(255.0)  # (N, 28, 28) in [0, 1]
    y = ds.targets.clone()
    return x, y


def weight_thresholds(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(10, 2) per-class ink terciles: thin < t0 <= regular < t1 <= bold. Ink = sum of [0,1] intensities."""
    ink = images.flatten(1).sum(1)
    th = torch.empty(10, 2)
    for c in range(10):
        th[c] = torch.quantile(ink[labels == c], torch.tensor([1 / 3, 2 / 3]))
    return th


def weight_of(ink: torch.Tensor, cls: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    t = th.to(ink.device)[cls]
    return (ink >= t[:, 0]).long() + (ink >= t[:, 1]).long()


class TwoDigitSampler:
    """Draws (image, prompt) batches by pasting two MNIST digits of the requested class and weight."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, device="cpu"):
        self.images = images.to(device)
        self.th = weight_thresholds(images, labels)
        w = weight_of(images.flatten(1).sum(1), labels, self.th)
        self.pool = [[torch.nonzero((labels == c) & (w == s)).flatten().to(device) for s in range(3)] for c in range(10)]
        self.device = device

    def draw_half(self, cls: torch.Tensor, wt: torch.Tensor, generator=None) -> torch.Tensor:
        idx = torch.empty_like(cls)
        key = cls * 3 + wt
        for k in torch.unique(key).tolist():
            m = key == k
            pool = self.pool[k // 3][k % 3]
            idx[m] = pool[torch.randint(len(pool), (int(m.sum()),), device=self.device, generator=generator)]
        return self.images[idx]

    def batch(self, n: int, prompts=None, generator=None):
        if prompts is None:
            pr = torch.stack([torch.randint(hi, (n,), device=self.device, generator=generator) for hi in (10, 3, 10, 3)], 1)
        else:
            P = torch.tensor(prompts, device=self.device)
            pr = P[torch.randint(len(P), (n,), device=self.device, generator=generator)]
        img = torch.cat([self.draw_half(pr[:, 0], pr[:, 1], generator), self.draw_half(pr[:, 2], pr[:, 3], generator)], -1)
        return img[:, None] * 2 - 1, pr
