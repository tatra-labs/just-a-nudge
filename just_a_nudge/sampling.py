"""Rectified-flow sampling: CFG with separate conditional/unconditional passes, ODE (Euler) and the
Flow-GRPO SDE step with its Gaussian log-likelihood.

Convention: x_t = (1 - t) x_0 + t * eps, so t = 1 is pure noise and the model predicts v = eps - x_0.
"""
from __future__ import annotations

import math

import torch

from .model import null_tokens

EVAL_STEPS = 20
TRAIN_STEPS = 10
# Step-bin boundary: the eval schedule's 20 noise levels split equally into early (t >= TAU) and late.
TAU = 0.525


def timesteps(n: int, shift: float = 1.0) -> list[float]:
    s = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    t = shift * s / (1 + (shift - 1) * s)
    return [float(v) for v in t]


def step_bin(t: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    return (t < tau).long()  # 0 = early, 1 = late


def model_velocity(model, x, t, tokens, pass_idx, steer=None, tau: float = TAU, **kw):
    """One pass of the network. t: (B,) tensor; pass_idx: (B,) 0 = conditional, 1 = unconditional."""
    sv = None
    if steer is not None:
        sv = steer.get(pass_idx, step_bin(t, tau))
    return model(x, t, tokens, steer=sv, **kw)


def cfg_velocity(model, x, t, tokens, w: float, steer=None, tau: float = TAU, return_passes: bool = False):
    B = x.shape[0]
    if not torch.is_tensor(t):
        t = torch.full((B,), float(t), device=x.device)
    xx = torch.cat([x, x])
    tt = torch.cat([t, t])
    tok = torch.cat([tokens, null_tokens(B, x.device)])
    p = torch.cat([torch.zeros(B, dtype=torch.long, device=x.device), torch.ones(B, dtype=torch.long, device=x.device)])
    v = model_velocity(model, xx, tt, tok, p, steer, tau)
    vc, vu = v.chunk(2)
    out = vu + w * (vc - vu)
    return (out, vc, vu) if return_passes else out


@torch.no_grad()
def sample_ode(model, tokens, w: float, noise: torch.Tensor, n_steps: int = EVAL_STEPS, steer=None,
               shift: float = 1.0, return_traj: bool = False):
    x = noise
    ts = timesteps(n_steps, shift)
    traj = []
    for i in range(n_steps):
        if return_traj:
            traj.append(x)
        v = cfg_velocity(model, x, ts[i], tokens, w, steer)
        x = x + (ts[i + 1] - ts[i]) * v
    return (x, traj, ts) if return_traj else x


def sde_mean_std(x, v, t: float, t_next: float, t_cap: float, eta: float):
    """Flow-GRPO's SDE update (mean and std of x_{t_next} | x_t)."""
    denom_sigma = t if t < 1 else t_cap
    std = eta * math.sqrt(t / (1 - denom_sigma))
    dt = t_next - t  # negative
    mean = x * (1 + std**2 / (2 * t) * dt) + v * (1 + std**2 * (1 - t) / (2 * t)) * dt
    return mean, std * math.sqrt(-dt)


def gauss_logp(x, mean, s: float):
    """Per-sample mean (over pixels) Gaussian log-density, as in Flow-GRPO."""
    lp = -((x - mean) ** 2) / (2 * s * s) - math.log(s) - 0.5 * math.log(2 * math.pi)
    return lp.mean(dim=tuple(range(1, x.dim())))
