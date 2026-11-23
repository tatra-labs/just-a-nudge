"""Paired evaluation on held-out prompts, and the paper's gain-share statistics.

Every policy is sampled from the same starting noise per (prompt, seed) with the deterministic ODE
sampler (20 Euler steps, CFG), so gains are paired. A gain is a policy's mean reward minus the base
model's; a share is a gain divided by the RL-tuned model's gain (the paper's "recovery"), with a paired
95% bootstrap interval over prompts (2,000 resamples).
"""
from __future__ import annotations

import numpy as np
import torch

from .lora import set_lora
from .model import null_tokens, prompt_tokens
from .rewards import both_correct, exact, ink, legible, match, upright, weight_correct
from .sampling import EVAL_STEPS, cfg_velocity, step_bin, timesteps


class EvalSet:
    def __init__(self, prompts, per_prompt: int, seed: int = 2026, device="cpu"):
        pr = torch.tensor(prompts, device=device).repeat_interleave(per_prompt, 0)
        self.prompt = pr
        self.prompt_id = torch.arange(len(prompts), device=device).repeat_interleave(per_prompt).cpu().numpy()
        g = torch.Generator(device=device).manual_seed(seed)
        self.noise = torch.randn(len(pr), 1, 28, 56, device=device, generator=g)
        self.n = len(pr)
        self.first_seed = np.zeros(self.n, dtype=bool)
        self.first_seed[::per_prompt] = True


def make_vfn(model, steer=None, lora: bool | None = None, w: float = 4.5, kv_reset: bool = False):
    """CFG velocity function x, t, tokens -> v for a policy (base model, optional LoRA, optional vectors).

    kv_reset: image queries read the text keys/values of the *unsteered base* on the same noisy state
    (the paper's Sec. 5 attention reset)."""

    def vfn(x, t, tok):
        if lora is not None:
            set_lora(model, lora)
        if not kv_reset:
            return cfg_velocity(model, x, t, tok, w, steer)
        B = x.shape[0]
        tt = torch.full((2 * B,), float(t), device=x.device)
        xx = torch.cat([x, x])
        tk = torch.cat([tok, null_tokens(B, x.device)])
        p = torch.cat([torch.zeros(B, dtype=torch.long, device=x.device), torch.ones(B, dtype=torch.long, device=x.device)])
        rec = {}
        if lora is not None:
            set_lora(model, False)
        model(xx, tt, tk, record=rec)
        if lora is not None:
            set_lora(model, lora)
        sv = steer.get(p, step_bin(tt)) if steer is not None else None
        v = model(xx, tt, tk, steer=sv, txt_kv_for_img=rec["txt_kv"])
        vc, vu = v.chunk(2)
        return vu + w * (vc - vu)

    return vfn


@torch.no_grad()
def sample_with(vfn, tokens, noise, n_steps: int = EVAL_STEPS):
    ts = timesteps(n_steps)
    x = noise
    for i in range(n_steps):
        x = x + (ts[i + 1] - ts[i]) * vfn(x, ts[i], tokens)
    return x


@torch.no_grad()
def evaluate(vfn, es: EvalSet, clfA, clfB, th, batch: int = 600, keep_first_seed: bool = False):
    """Rewards per sample. keep_first_seed: also return the image of every prompt's first seed."""
    out = {k: [] for k in ("match", "exact_B", "weight_ok", "legible", "correct_B", "upright", "ink")}
    imgs = []
    for i in range(0, es.n, batch):
        pr = es.prompt[i : i + batch]
        x = sample_with(vfn, prompt_tokens(pr), es.noise[i : i + batch])
        out["match"].append(match(clfA, x, pr, th))
        out["exact_B"].append(exact(clfB, x, pr, th))
        out["weight_ok"].append(weight_correct(x, pr, th))
        out["legible"].append(legible(clfA, x, pr))
        out["correct_B"].append(both_correct(clfB, x, pr))
        out["upright"].append(upright(x))
        out["ink"].append(ink(x))
        if keep_first_seed:
            imgs.append(x[torch.from_numpy(es.first_seed[i : i + batch]).to(x.device)].cpu())
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    if keep_first_seed:
        res["images"] = torch.cat(imgs)
    return res


def per_prompt(values: np.ndarray, prompt_id: np.ndarray) -> np.ndarray:
    P = prompt_id.max() + 1
    return np.bincount(prompt_id, weights=values, minlength=P) / np.bincount(prompt_id, minlength=P)


def gain_stats(base: np.ndarray, other: np.ndarray, prompt_id: np.ndarray, n_boot: int = 2000, seed: int = 0):
    g = per_prompt(other - base, prompt_id)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    b = g[idx].mean(1)
    return {"gain": float(g.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def share_stats(base: np.ndarray, rl: np.ndarray, other: np.ndarray, prompt_id: np.ndarray, n_boot: int = 2000,
                seed: int = 0):
    g_rl = per_prompt(rl - base, prompt_id)
    g_o = per_prompt(other - base, prompt_id)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g_rl), size=(n_boot, len(g_rl)))
    shares = 100 * g_o[idx].mean(1) / g_rl[idx].mean(1)
    return {
        "share": float(100 * g_o.mean() / g_rl.mean()),
        "ci": [float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))],
    }


def summarize(res: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in res.items() if k != "images"}
