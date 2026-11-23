"""Fitting steering vectors (or a small LoRA) to an RL-tuned model, with no reward in the loop.

* `velocity_match` - the paper's Eq. 2: regress the steered base's per-pass velocity onto the RL-tuned
  model's, on the RL model's own final samples re-noised to one of the sampler's noise levels
  (fresh noise and level every update, "new states").
* `mean_difference` - the paper's s_mean: for each block in order, the RL model's block output minus
  the steered base's, averaged over tokens, prompts and the noise levels of each step bin, on the same
  states along the RL model's sampling trajectories, with earlier blocks' vectors already added.
* `random_like` - random directions rescaled to the fitted vectors' per-vector norms.
"""
from __future__ import annotations

import time

import torch

from .model import null_tokens, prompt_tokens
from .sampling import EVAL_STEPS, model_velocity, sample_ode, step_bin, timesteps


@torch.no_grad()
def collect_samples(vel_model, steer, prompts, per_prompt: int, w: float, seed: int = 123, batch: int = 512,
                    keep_traj: int = 0, before=None):
    """Final ODE samples (and optionally trajectories) of a policy on the given prompts.

    `before` is called before every batch (e.g. to switch a LoRA on)."""
    dev = next(vel_model.parameters()).device
    pr = torch.tensor(prompts, device=dev).repeat(per_prompt, 1)  # tiled: any prefix covers every prompt
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn(len(pr), 1, 28, 56, device=dev, generator=g)
    xs, trajs = [], []
    for i in range(0, len(pr), batch):
        if before is not None:
            before()
        tok = prompt_tokens(pr[i : i + batch])
        x, traj, ts = sample_ode(vel_model, tok, w, noise[i : i + batch], steer=steer, return_traj=True)
        xs.append(x)
        if keep_traj and len(trajs) * batch < keep_traj:
            trajs.append(torch.stack(traj, 1))  # (b, steps, 1, 28, 56)
    out = {"x0": torch.cat(xs), "prompt": pr}
    if keep_traj:
        out["traj"] = torch.cat(trajs)[:keep_traj]
        out["traj_prompt"] = pr[: len(out["traj"])]
        out["ts"] = timesteps(EVAL_STEPS)
    return out


def _lr_lambda(updates: int, warm: int = 50):
    def f(s):
        if s < warm:
            return (s + 1) / warm
        return 1.0 - 0.9 * (s - warm) / max(1, updates - warm)

    return f


def velocity_match(student_fn, params, teacher_fn, data, updates: int = 1500, batch: int = 64, lr: float = 3e-3,
                   seed: int = 0, log=print, n_levels: int = EVAL_STEPS):
    """student_fn / teacher_fn: (x, t, tokens, pass_idx) -> velocity. Both passes in every update."""
    dev = data["x0"].device
    g = torch.Generator(device=dev).manual_seed(seed)
    ts = torch.tensor(timesteps(n_levels)[:-1], device=dev, dtype=torch.float32)  # the sampler's noise levels
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(updates))
    N = len(data["x0"])
    t0 = time.time()
    hist = []
    for u in range(updates):
        idx = torch.randint(N, (batch,), device=dev, generator=g)
        x0 = data["x0"][idx]
        t = ts[torch.randint(len(ts), (batch,), device=dev, generator=g)]
        eps = torch.randn(x0.shape, device=dev, generator=g)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
        tok = prompt_tokens(data["prompt"][idx])
        xx = torch.cat([xt, xt])
        tt = torch.cat([t, t])
        tk = torch.cat([tok, null_tokens(batch, dev)])
        p = torch.cat([torch.zeros(batch, dtype=torch.long, device=dev), torch.ones(batch, dtype=torch.long, device=dev)])
        with torch.no_grad():
            target = teacher_fn(xx, tt, tk, p)
        loss = ((student_fn(xx, tt, tk, p) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
        if u % 250 == 0 or u == updates - 1:
            log(f"  velocity matching {u}/{updates} loss {loss.item():.6f} ({time.time() - t0:.0f}s)")
    return hist


@torch.no_grad()
def velocity_gap(student_fn, teacher_fn, base_fn, data, n: int = 1024, seed: int = 7):
    """Fraction of the base-to-teacher velocity gap closed by the student (1 - ||s-T||^2 / ||b-T||^2)."""
    dev = data["x0"].device
    g = torch.Generator(device=dev).manual_seed(seed)
    ts = torch.tensor(timesteps(EVAL_STEPS)[:-1], device=dev, dtype=torch.float32)
    idx = torch.randint(len(data["x0"]), (n,), device=dev, generator=g)
    x0 = data["x0"][idx]
    t = ts[torch.randint(len(ts), (n,), device=dev, generator=g)]
    eps = torch.randn(x0.shape, device=dev, generator=g)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
    tok = prompt_tokens(data["prompt"][idx])
    out = {}
    for name, tk, pidx in (("cond", tok, 0), ("uncond", null_tokens(n, dev), 1)):
        p = torch.full((n,), pidx, dtype=torch.long, device=dev)
        T = teacher_fn(xt, t, tk, p)
        S = student_fn(xt, t, tk, p)
        B0 = base_fn(xt, t, tk, p)
        out[name] = 1.0 - ((S - T) ** 2).sum().item() / ((B0 - T) ** 2).sum().item()
    return out


@torch.no_grad()
def mean_difference(base_model, steer, teacher_record_fn, data, batch: int = 256):
    """Sequential per-block mean difference (s_mean). `teacher_record_fn(x,t,tok,p,record)` runs the RL model.

    Uses the stored ODE trajectories of the RL model (`data['traj']`, (N, steps, 1, 28, 56))."""
    traj = data["traj"]
    N, S = traj.shape[:2]
    dev = traj.device
    ts = torch.tensor(data["ts"][:-1], device=dev, dtype=torch.float32)
    steer.img.zero_()
    steer.txt.zero_()
    L = steer.img.shape[2]
    for l in range(L):
        sums = {k: torch.zeros_like(getattr(steer, k)[:, :, 0]) for k in ("img", "txt")}
        counts = torch.zeros(steer.img.shape[0], steer.img.shape[1], device=dev)
        for i in range(0, N, batch):
            xb = traj[i : i + batch]
            nb = len(xb)
            tok = prompt_tokens(data["traj_prompt"][i : i + batch])
            for s in range(S):
                x = xb[:, s]
                t = ts[s].expand(nb)
                for pidx, tk in ((0, tok), (1, null_tokens(nb, dev))):
                    p = torch.full((nb,), pidx, dtype=torch.long, device=dev)
                    rec_t, rec_b = {}, {}
                    teacher_record_fn(x, t, tk, p, rec_t)
                    model_velocity(base_model, x, t, tk, p, steer, record=rec_b)
                    k = int(step_bin(t[:1])[0])
                    sums["img"][pidx, k] += (rec_t["img_out"][l] - rec_b["img_out"][l]).mean(1).sum(0)
                    if l < steer.txt.shape[2]:
                        sums["txt"][pidx, k] += (rec_t["txt_out"][l] - rec_b["txt_out"][l]).mean(1).sum(0)
                    counts[pidx, k] += nb
        steer.img[:, :, l] = sums["img"] / counts[..., None]
        if l < steer.txt.shape[2]:
            steer.txt[:, :, l] = sums["txt"] / counts[..., None]
    return steer


@torch.no_grad()
def random_like(steer_src, steer_dst, seed: int = 0):
    g = torch.Generator(device=steer_src.img.device).manual_seed(seed)
    for k in ("img", "txt"):
        src = getattr(steer_src, k)
        r = torch.randn(src.shape, device=src.device, generator=g)
        r = r / r.norm(dim=-1, keepdim=True) * src.norm(dim=-1, keepdim=True)
        getattr(steer_dst, k).copy_(r)
    return steer_dst


def vector_norm_stats(steer) -> dict:
    n_img = steer.img.norm(dim=-1)
    n_txt = steer.txt.norm(dim=-1)
    return {
        "img_norm_mean": n_img.mean().item(),
        "txt_norm_mean": n_txt.mean().item(),
        "max_norm": max(n_img.max().item(), n_txt.max().item()),
        "cos_cond_uncond_mean": torch.cat(
            [
                torch.nn.functional.cosine_similarity(steer.img[0], steer.img[1], dim=-1).flatten(),
                torch.nn.functional.cosine_similarity(steer.txt[0], steer.txt[1], dim=-1).flatten(),
            ]
        ).mean().item(),
    }

