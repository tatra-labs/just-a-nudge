"""Stage 2, per reward: RL-tune with Flow-GRPO, fit steering vectors, run every control, evaluate.

  python scripts/02_run.py --reward match
  python scripts/02_run.py --reward upright

Artifacts are cached in checkpoints/<reward>/ (delete a file to recompute that step).
Results go to results/<reward>.json and results/<reward>_samples.pt.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from just_a_nudge.data import prompt_split  # noqa: E402
from just_a_nudge.evaluate import EvalSet, evaluate, gain_stats, make_vfn, share_stats, summarize  # noqa: E402
from just_a_nudge.fit import (collect_samples, mean_difference, random_like, vector_norm_stats,  # noqa: E402
                              velocity_gap, velocity_match)
from just_a_nudge.grpo import GRPOConfig, Policy, train_grpo  # noqa: E402
from just_a_nudge.io import CKPT, RESULTS, Logger, load_base, load_classifier, write_json  # noqa: E402
from just_a_nudge.lora import ATTN_TARGETS, OUT_TARGETS, inject_lora, lora_params, lora_state, set_lora  # noqa: E402
from just_a_nudge.rewards import match, upright  # noqa: E402
from just_a_nudge.sampling import model_velocity  # noqa: E402
from just_a_nudge.steering import COND, EARLY, LATE, UNCOND, SteeringVectors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reward", choices=["match", "upright"], required=True)
ap.add_argument("--tag", default="", help="suffix for a variant run (e.g. nokl)")
ap.add_argument("--base", default="base.pt", help="base checkpoint under checkpoints/")
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--rl-updates", type=int, default=300)
ap.add_argument("--rl-lr", type=float, default=3e-4)
ap.add_argument("--lora-rank", type=int, default=32)
ap.add_argument("--beta-kl", type=float, default=0.04)
ap.add_argument("--steer-rl-updates", type=int, default=300)
ap.add_argument("--steer-rl-lr", type=float, default=5e-3)
ap.add_argument("--fit-updates", type=int, default=3000)
ap.add_argument("--fit-lr", type=float, default=3e-3)
ap.add_argument("--cache-per-prompt", type=int, default=4)
ap.add_argument("--eval-per-prompt", type=int, default=8)
ap.add_argument("--skip-steer-rl", action="store_true")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
name = args.reward + (f"_{args.tag}" if args.tag else "")
out = CKPT / name
out.mkdir(parents=True, exist_ok=True)
log = Logger(Path(__file__).resolve().parent.parent / "logs" / f"02_{name}.log")
log(f"=== {name} {vars(args)}")
t_start = time.time()

base = load_base(dev, CKPT / args.base)
clfA, clfB = load_classifier("A", dev), load_classifier("B", dev)
train_prompts, test_prompts = prompt_split()
th = torch.load(CKPT / "weight_thresholds.pt", weights_only=True)
reward_fn = {
    "match": lambda x, pr: match(clfA, x, pr, th),
    "upright": lambda x, pr: upright(x),
}[args.reward]
es = EvalSet(test_prompts, per_prompt=args.eval_per_prompt, device=dev)
mini = EvalSet(test_prompts, per_prompt=1, seed=99, device=dev)


def quick_eval(vfn):
    s = summarize(evaluate(vfn, mini, clfA, clfB, th))
    return {k: round(v, 4) for k, v in s.items()}


# ---------------------------------------------------------------- RL-tuned checkpoint (LoRA)
rl = copy.deepcopy(base)
inject_lora(rl, ATTN_TARGETS, rank=args.lora_rank)
rl.to(dev)
rl_params = lora_params(rl)
for p in rl_params:
    p.requires_grad_(True)
rl_cfg = GRPOConfig(reward=args.reward, updates=args.rl_updates, cfg=args.cfg, lr=args.rl_lr, beta_kl=args.beta_kl)
f = out / "rl_lora.pt"
if f.exists():
    ck = torch.load(f, map_location=dev, weights_only=False)
    rl.load_state_dict(ck["state"], strict=False)
    rl_hist = ck["history"]
    log("loaded RL LoRA")
else:
    rl_hist = train_grpo(Policy(rl, has_lora=True), rl_params, reward_fn, train_prompts, rl_cfg, log=log,
                         eval_fn=lambda: quick_eval(make_vfn(rl, lora=True, w=args.cfg)))
    torch.save({"state": lora_state(rl), "history": rl_hist, "cfg": rl_cfg.to_dict()}, f)
set_lora(rl, True)
for p in rl_params:
    p.requires_grad_(False)

teacher_fn = lambda x, t, tok, p: model_velocity(rl, x, t, tok, p)  # noqa: E731
teacher_rec = lambda x, t, tok, p, rec: model_velocity(rl, x, t, tok, p, record=rec)  # noqa: E731
base_fn = lambda x, t, tok, p: model_velocity(base, x, t, tok, p)  # noqa: E731

# ---------------------------------------------------------------- the RL model's own samples (no reward)
f = out / "rl_samples.pt"
if f.exists():
    data = torch.load(f, map_location=dev, weights_only=False)
else:
    data = collect_samples(rl, None, train_prompts, args.cache_per_prompt, args.cfg, keep_traj=256)
    torch.save(data, f)
log(f"RL sample cache: {len(data['x0'])} final samples, {len(data['traj'])} trajectories")


def load_or_fit_steer(fname, fit_fn):
    s = SteeringVectors.for_model(base).to(dev)
    p = out / fname
    if p.exists():
        s.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
        return s, None
    hist = fit_fn(s)
    torch.save(s.state_dict(), p)
    return s, hist


# ---------------------------------------------------------------- s_fit: velocity matching (Eq. 2)
def fit_vm(s):
    return velocity_match(lambda x, t, tok, p: model_velocity(base, x, t, tok, p, s), [s.img, s.txt], teacher_fn,
                          data, updates=args.fit_updates, batch=64, lr=args.fit_lr, log=log)


s_fit, fit_hist = load_or_fit_steer("steer_fit.pt", fit_vm)
for p in s_fit.parameters():
    p.requires_grad_(False)
log("s_fit ready")

# ---------------------------------------------------------------- s_mean, random control
s_mean, _ = load_or_fit_steer("steer_mean.pt", lambda s: mean_difference(base, s, teacher_rec, data))
s_rand = random_like(s_fit, SteeringVectors.for_model(base).to(dev))
log("s_mean / s_rand ready")

# ---------------------------------------------------------------- same-size LoRA control, fitted the same way
lo = copy.deepcopy(base)
inject_lora(lo, OUT_TARGETS, rank=2)
lo.to(dev)
lo_params = lora_params(lo)
f = out / "lora2_fit.pt"
if f.exists():
    lo.load_state_dict(torch.load(f, map_location=dev, weights_only=True), strict=False)
else:
    for p in lo_params:
        p.requires_grad_(True)
    velocity_match(lambda x, t, tok, p: model_velocity(lo, x, t, tok, p), lo_params, teacher_fn, data,
                   updates=args.fit_updates, batch=64, lr=args.fit_lr, log=log)
    torch.save(lora_state(lo), f)
for p in lo_params:
    p.requires_grad_(False)
set_lora(lo, True)
log("same-size LoRA ready")

# ---------------------------------------------------------------- s_RL: Flow-GRPO on the vectors only
s_rl, s_rl_hist = None, None
if not args.skip_steer_rl:
    s_rl = SteeringVectors.for_model(base).to(dev)
    f = out / "steer_rl.pt"
    if f.exists():
        ck = torch.load(f, map_location=dev, weights_only=False)
        s_rl.load_state_dict(ck["state"])
        s_rl_hist = ck["history"]
    else:
        cfg2 = GRPOConfig(reward=args.reward, updates=args.steer_rl_updates, cfg=args.cfg, lr=args.steer_rl_lr,
                          beta_kl=args.beta_kl)
        s_rl_hist = train_grpo(Policy(base, s_rl), [s_rl.img, s_rl.txt], reward_fn, train_prompts, cfg2, log=log,
                               eval_fn=lambda: quick_eval(make_vfn(base, s_rl, w=args.cfg)))
        torch.save({"state": s_rl.state_dict(), "history": s_rl_hist, "cfg": cfg2.to_dict()}, f)
    for p in s_rl.parameters():
        p.requires_grad_(False)

# ---------------------------------------------------------------- velocity-gap closure on held-in states
gaps = {
    "s_fit": velocity_gap(lambda x, t, tok, p: model_velocity(base, x, t, tok, p, s_fit), teacher_fn, base_fn, data),
    "s_mean": velocity_gap(lambda x, t, tok, p: model_velocity(base, x, t, tok, p, s_mean), teacher_fn, base_fn, data),
    "lora2": velocity_gap(lambda x, t, tok, p: model_velocity(lo, x, t, tok, p), teacher_fn, base_fn, data),
}
log(f"velocity gap closed: {gaps}")

# ---------------------------------------------------------------- evaluation
policies = {
    "base": make_vfn(base, w=args.cfg),
    "rl_lora": make_vfn(rl, lora=True, w=args.cfg),
    "s_fit": make_vfn(base, s_fit, w=args.cfg),
    "s_mean": make_vfn(base, s_mean, w=args.cfg),
    "s_rand": make_vfn(base, s_rand, w=args.cfg),
    "lora2_fit": make_vfn(lo, lora=True, w=args.cfg),
}
if s_rl is not None:
    policies["s_rl"] = make_vfn(base, s_rl, w=args.cfg)

ablations = {
    "cond_only": dict(passes=[COND]),
    "uncond_only": dict(passes=[UNCOND]),
    "early_only": dict(bins=[EARLY]),
    "late_only": dict(bins=[LATE]),
    "text_only": dict(streams=["txt"]),
    "image_only": dict(streams=["img"]),
    "cond_text_early": dict(passes=[COND], streams=["txt"], bins=[EARLY]),
}
L = base.cfg.depth
half = L // 2
ablations[f"blocks_0-{half - 1}"] = dict(blocks=list(range(half)))
ablations[f"blocks_{half}-{L - 1}"] = dict(blocks=list(range(half, L)))
abl_steers = {}
for k, sel in ablations.items():
    s = copy.deepcopy(s_fit)
    s.keep_only(**sel)
    abl_steers[k] = s
    policies[f"abl_{k}"] = make_vfn(base, s, w=args.cfg)

cfg_grid = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0]
for w in cfg_grid:
    policies[f"cfg_{w:g}"] = make_vfn(base, w=w)
policies["kvreset_rl_lora"] = make_vfn(rl, lora=True, w=args.cfg, kv_reset=True)
policies["kvreset_s_fit"] = make_vfn(base, s_fit, w=args.cfg, kv_reset=True)

evals = {}
for k, vfn in policies.items():
    evals[k] = evaluate(vfn, es, clfA, clfB, th, keep_first_seed=True)
    log(f"eval {k}: {summarize(evals[k])}")

metric = args.reward
METRICS = ("match", "exact_B", "weight_ok", "legible", "correct_B", "upright", "ink")
pid = es.prompt_id
res = {
    "reward": args.reward,
    "args": vars(args),
    "rl_config": rl_cfg.to_dict(),
    "n_train_prompts": len(train_prompts),
    "test_prompts": test_prompts,
    "eval_samples": int(es.n),
    "params": {
        "steering_vectors": s_fit.numel(),
        "n_vectors": s_fit.n_vectors(),
        "rl_lora": sum(p.numel() for p in rl_params),
        "same_size_lora": sum(p.numel() for p in lo_params),
        "base": sum(p.numel() for p in base.parameters()),
    },
    "velocity_gap_closed": gaps,
    "vector_norms": {"s_fit": vector_norm_stats(s_fit), "s_mean": vector_norm_stats(s_mean)},
    "summary": {k: summarize(v) for k, v in evals.items()},
    "gain": {k: gain_stats(evals["base"][metric], v[metric], pid) for k, v in evals.items() if k != "base"},
    "share": {k: share_stats(evals["base"][metric], evals["rl_lora"][metric], v[metric], pid)
              for k, v in evals.items() if k not in ("base", "rl_lora")},
    "side_effects": {
        k: {m: gain_stats(evals["base"][m], evals[k][m], pid) for m in METRICS}
        for k in ("rl_lora", "s_fit", "s_rl") if k in evals
    },
    "history": {
        "rl_lora": rl_hist,
        "s_rl": s_rl_hist,
        "fit_loss_every_50": fit_hist[::50] if fit_hist else None,
    },
    "minutes": (time.time() - t_start) / 60,
}
res["params"]["lora_over_vectors"] = res["params"]["rl_lora"] / res["params"]["steering_vectors"]
best_cfg = max(cfg_grid, key=lambda w: res["share"][f"cfg_{w:g}"]["share"])
res["best_cfg_control"] = {"cfg": best_cfg, **res["share"][f"cfg_{best_cfg:g}"]}
write_json(res, f"{name}.json")
FIG_KEYS = ("base", "rl_lora", "s_fit", "s_rl", "s_mean", "lora2_fit", "s_rand", "cfg_8")
torch.save({k: v["images"].half() for k, v in evals.items() if k in FIG_KEYS} | {"test_prompts": test_prompts},
           RESULTS / f"{name}_samples.pt")
log(f"shares: { {k: round(v['share'], 1) for k, v in res['share'].items()} }")
log(f"done in {res['minutes']:.1f} min")
