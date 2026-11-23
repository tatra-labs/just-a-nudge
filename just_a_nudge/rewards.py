"""The two rewards, mirroring the paper's two kinds.

* `match` (GenEval / OCR analogue, verifiable and prompt-dependent): for each half of the image,
  a frozen MNIST classifier's probability of the prompt's digit times a soft check that the digit's
  ink lies in the prompt's stroke-weight tercile for that class; the reward is the mean over the two
  halves. `exact` is the hard version (both digits right per an independently trained classifier B
  with a different architecture, and both weights in range), reported but never optimised.
* `upright` (PickScore analogue, dense and prompt-independent): minus the mean absolute slant of the
  two digits, measured from image moments (the classic deskew shear mu11 / mu02). A small penalty
  applies when a digit has less than half of MNIST's typical ink, so blank halves cannot score.
  No pixel-wise tone curve can change a slant, unlike a brightness or contrast reward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INK_REF = 100.0  # MNIST's mean total intensity of a 28x28 digit is ~103


class ClassifierA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.f1 = nn.Linear(64 * 12 * 12, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):  # x in [0, 1], (B,1,28,28)
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


class ClassifierB(nn.Module):
    """A different architecture (wider kernels, no pooling until late), for evaluation only."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5, padding=2)
        self.c2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.c3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.f = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = F.gelu(self.c1(x))
        x = F.gelu(self.c2(x))
        x = F.gelu(self.c3(x))
        return self.f(x.flatten(1))


def halves(img: torch.Tensor):
    """(B,1,28,56) in [-1,1] -> two (B,1,28,28) tensors in [0,1]."""
    x = ((img + 1) / 2).clamp(0, 1)
    return x[..., :28], x[..., 28:]


def train_classifier(model: nn.Module, images, labels, epochs: int = 3, device="cpu", seed: int = 0, log=print):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = images[:, None].to(device)
    Y = labels.to(device)
    n = len(X)
    steps = epochs * (n // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for step in range(steps):
        idx = torch.randint(n, (128,), device=device)
        x = X[idx]
        # light augmentation: random shifts up to 2 px and additive noise like a sampler leaves
        dx, dy = torch.randint(-2, 3, (2,)).tolist()
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        x = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
        loss = F.cross_entropy(model(x), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            log(f"  classifier step {step}/{steps} loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, images, labels, device="cpu") -> float:
    correct = 0
    for i in range(0, len(images), 2048):
        x = images[i : i + 2048, None].to(device)
        correct += (model(x).argmax(-1).cpu() == labels[i : i + 2048]).sum().item()
    return correct / len(images)


def _ink(x):
    return x.flatten(1).sum(1)


def legible(clf: nn.Module, img: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    """Mean classifier probability of the two prompted digits. pr: (B, 4) prompts."""
    a, b = halves(img)
    pa = F.softmax(clf(a), -1).gather(1, pr[:, 0:1]).squeeze(1)
    pb = F.softmax(clf(b), -1).gather(1, pr[:, 2:3]).squeeze(1)
    return 0.5 * (pa + pb)


def both_correct(clf: nn.Module, img: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    a, b = halves(img)
    return ((clf(a).argmax(-1) == pr[:, 0]) & (clf(b).argmax(-1) == pr[:, 2])).float()


WEIGHT_TAU = 3.0  # ink units; the regular tercile is ~20-28 units wide


def soft_weight(ink: torch.Tensor, cls: torch.Tensor, wt: torch.Tensor, th: torch.Tensor, tau: float = WEIGHT_TAU):
    """Soft indicator that `ink` lies in tercile `wt` of class `cls` (th: (10, 2) thresholds)."""
    t = th.to(ink.device)[cls]
    above0 = torch.sigmoid((ink - t[:, 0]) / tau)
    above1 = torch.sigmoid((ink - t[:, 1]) / tau)
    probs = torch.stack([1 - above0, above0 - above1, above1], 1)  # a soft partition (t0 < t1)
    return probs.gather(1, wt[:, None]).squeeze(1)


def hard_weight(ink: torch.Tensor, cls: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    t = th.to(ink.device)[cls]
    return (ink >= t[:, 0]).long() + (ink >= t[:, 1]).long()


def match(clf: nn.Module, img: torch.Tensor, pr: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """The verifiable reward: mean over halves of p(digit) * soft weight check."""
    out = 0.0
    for h, (ci, wi) in zip(halves(img), ((0, 1), (2, 3))):
        p = F.softmax(clf(h), -1).gather(1, pr[:, ci : ci + 1]).squeeze(1)
        out = out + 0.5 * p * soft_weight(_ink(h), pr[:, ci], pr[:, wi], th)
    return out


def weight_correct(img: torch.Tensor, pr: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """Fraction of the two halves whose ink is in the prompted tercile (of the prompted class)."""
    a, b = halves(img)
    return 0.5 * ((hard_weight(_ink(a), pr[:, 0], th) == pr[:, 1]).float() + (hard_weight(_ink(b), pr[:, 2], th) == pr[:, 3]).float())


def exact(clf: nn.Module, img: torch.Tensor, pr: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """1 if both digits are right (per `clf`) and both stroke weights are in range."""
    a, b = halves(img)
    ok = (clf(a).argmax(-1) == pr[:, 0]) & (clf(b).argmax(-1) == pr[:, 2])
    ok &= (hard_weight(_ink(a), pr[:, 0], th) == pr[:, 1]) & (hard_weight(_ink(b), pr[:, 2], th) == pr[:, 3])
    return ok.float()


def slant(x: torch.Tensor):
    """Deskew shear mu11/mu02 and total ink of (B,1,28,28) images in [0,1]."""
    B = x.shape[0]
    I = x.reshape(B, 28, 28)
    ys = torch.arange(28, device=x.device, dtype=x.dtype)[:, None].expand(28, 28)
    xs = torch.arange(28, device=x.device, dtype=x.dtype)[None, :].expand(28, 28)
    m = I.sum((1, 2)).clamp_min(1e-6)
    cy = (I * ys).sum((1, 2)) / m
    cx = (I * xs).sum((1, 2)) / m
    dy = ys[None] - cy[:, None, None]
    dx = xs[None] - cx[:, None, None]
    mu11 = (I * dx * dy).sum((1, 2)) / m
    mu02 = (I * dy * dy).sum((1, 2)) / m
    return mu11 / mu02.clamp_min(1.0), m


def upright(img: torch.Tensor) -> torch.Tensor:
    out = 0.0
    for h in halves(img):
        s, m = slant(h)
        out = out - 0.5 * s.abs() - 0.5 * 2.0 * F.relu(0.5 - m / INK_REF)
    return out


def ink(img: torch.Tensor) -> torch.Tensor:
    return sum(slant(h)[1] for h in halves(img)) / (2 * INK_REF)
