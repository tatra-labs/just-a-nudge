"""Flow-GRPO (Liu et al. 2025) for the tiny MM-DiT.

The policy is the CFG-combined velocity of the base model plus whatever is trainable: a LoRA on the
attention projections (the usual recipe; gives the "RL-tuned checkpoint") or only the steering vectors
on a frozen base (the paper's Sec. 4.6, "s_RL"). Exploration uses Flow-GRPO's SDE sampler; the
policy-gradient loss is the clipped PPO objective with group-relative advantages, plus an optional KL
to the reference (LoRA disabled / vectors off) computed in closed form between the Gaussian steps.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .lora import set_lora
from .model import prompt_tokens
from .sampling import TRAIN_STEPS, cfg_velocity, gauss_logp, sde_mean_std, timesteps


@dataclass
class GRPOConfig:
    reward: str = "match"
    updates: int = 300
    prompts_per_update: int = 16
    group: int = 8
    steps: int = TRAIN_STEPS
    eta: float = 0.7
    cfg: float = 4.5
    lr: float = 3e-4
    beta_kl: float = 0.04
    clip: float = 1e-4
    adv_clip: float = 5.0
    minibatches: int = 2
    seed: int = 0

    def to_dict(self):
        return asdict(self)


class Policy:
    """model (+ optional LoRA, toggled by flag) and optional steering vectors."""

    def __init__(self, model, steer=None, has_lora: bool = False):
        self.model = model
        self.steer = steer
        self.has_lora = has_lora

    def velocity(self, x, t, tokens, w):
        if self.has_lora:
            set_lora(self.model, True)
        return cfg_velocity(self.model, x, t, tokens, w, self.steer)

    @torch.no_grad()
    def ref_velocity(self, x, t, tokens, w):
        if self.has_lora:
            set_lora(self.model, False)
        v = cfg_velocity(self.model, x, t, tokens, w, None)
        if self.has_lora:
            set_lora(self.model, True)
        return v


@torch.no_grad()
def rollout(policy: Policy, tokens, cfg: GRPOConfig, generator=None):
    ts = timesteps(cfg.steps)
    B = tokens.shape[0]
    dev = tokens.device
    x = torch.randn(B, 1, 28, 56, device=dev, generator=generator)
    xs, prevs, logps = [], [], []
    for i in range(cfg.steps):
        v = policy.velocity(x, ts[i], tokens, cfg.cfg)
        mean, s = sde_mean_std(x, v, ts[i], ts[i + 1], ts[1], cfg.eta)
        prev = mean + s * torch.randn(x.shape, device=dev, generator=generator)
        xs.append(x)
        prevs.append(prev)
        logps.append(gauss_logp(prev, mean, s))
        x = prev
    # The reward sees the last step's mean, not its noisy draw: the final SDE step leaves per-pixel noise
    # (std ~0.07) that adds ~12 units of ink and shifts slant moments, so rewarding the noisy draw teaches
    # the policy to pre-compensate for noise that the ODE sampler used at evaluation never adds.
    return mean, xs, prevs, logps


def train_grpo(policy: Policy, params, reward_fn, train_prompts, cfg: GRPOConfig, log=print, eval_fn=None,
               eval_every: int = 50):
    dev = next(policy.model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    ts = timesteps(cfg.steps)
    prompts = torch.tensor(train_prompts, device=dev)
    history = []
    t0 = time.time()
    for u in range(cfg.updates):
        pick = prompts[torch.randperm(len(prompts), device=dev, generator=g)[: cfg.prompts_per_update]]
        pr = pick.repeat_interleave(cfg.group, dim=0)
        tok = prompt_tokens(pr)
        x0, xs, prevs, logps = rollout(policy, tok, cfg, generator=g)
        with torch.no_grad():
            r = reward_fn(x0, pr)
            rg = r.view(cfg.prompts_per_update, cfg.group)
            adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).flatten()
            adv = adv.clamp(-cfg.adv_clip, cfg.adv_clip)
        B = tok.shape[0]
        perm = torch.randperm(B, device=dev, generator=g)
        kl_acc = 0.0
        for mb in perm.chunk(cfg.minibatches):
            opt.zero_grad(set_to_none=True)
            for i in range(cfg.steps):
                v = policy.velocity(xs[i][mb], ts[i], tok[mb], cfg.cfg)
                mean, s = sde_mean_std(xs[i][mb], v, ts[i], ts[i + 1], ts[1], cfg.eta)
                logp = gauss_logp(prevs[i][mb], mean, s)
                ratio = torch.exp(logp - logps[i][mb])
                a = adv[mb]
                loss = torch.maximum(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                if cfg.beta_kl > 0:
                    v_ref = policy.ref_velocity(xs[i][mb], ts[i], tok[mb], cfg.cfg)
                    mean_ref, _ = sde_mean_std(xs[i][mb], v_ref, ts[i], ts[i + 1], ts[1], cfg.eta)
                    kl = ((mean - mean_ref) ** 2).mean(dim=(1, 2, 3)) / (2 * s * s)
                    loss = loss + cfg.beta_kl * kl.mean()
                    kl_acc += kl.mean().item() / (cfg.steps * cfg.minibatches)
                (loss / cfg.steps).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        rec = {"update": u, "reward": r.mean().item(), "kl": kl_acc}
        if eval_fn is not None and (u % eval_every == 0 or u == cfg.updates - 1):
            rec["eval"] = eval_fn()
        history.append(rec)
        if u % 10 == 0 or u == cfg.updates - 1 or "eval" in rec:
            log(f"[{cfg.reward}] update {u} reward {rec['reward']:.4f} kl {kl_acc:.5f} "
                f"{'eval ' + str(rec.get('eval')) if 'eval' in rec else ''} ({time.time() - t0:.0f}s)")
    return history
