"""Pretrain the base MM-DiT by rectified flow on all 900 prompts (10% condition dropout)."""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from .data import TwoDigitSampler, load_mnist
from .model import DiTConfig, MMDiT, null_tokens, prompt_tokens


def train_base(steps: int = 15000, batch: int = 256, lr: float = 5e-4, p_uncond: float = 0.1,
               ema_decay: float = 0.999, device: str = "cuda", seed: int = 0, cfg: DiTConfig = DiTConfig(),
               log=print, snapshots=(), on_snapshot=None):
    torch.manual_seed(seed)
    images, labels = load_mnist(train=True)
    sampler = TwoDigitSampler(images, labels, device=device)
    model = MMDiT(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * max(0.0, 1 - s / steps))
    )
    t0 = time.time()
    for step in range(steps):
        x0, pr = sampler.batch(batch)
        tok = prompt_tokens(pr)
        drop = torch.rand(batch, device=device) < p_uncond
        tok = torch.where(drop[:, None], null_tokens(batch, device), tok)
        t = torch.sigmoid(torch.randn(batch, device=device))  # logit-normal(0, 1), as SD3
        eps = torch.randn_like(x0)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
        loss = F.mse_loss(model(xt, t, tok), eps - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if (step + 1) in snapshots and on_snapshot is not None:
            on_snapshot(step + 1, ema)
        if step % 500 == 0 or step == steps - 1:
            log(f"base step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    return ema
