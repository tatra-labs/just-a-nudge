"""Stage 4: markdown tables (results/tables.md) from results/*.json, so the README quotes generated numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from just_a_nudge.io import RESULTS, read_json  # noqa: E402

ORDER = ["match", "upright", "match_weakbase", "upright_nokl"]
runs = [r for r in ORDER if (RESULTS / f"{r}.json").exists()]
L = []


def sh(R, k):
    s = R["share"].get(k)
    return "–" if s is None else f"{s['share']:.1f} [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]"


def g(R, k):
    s = R["gain"].get(k)
    return "–" if s is None else f"{s['gain']:+.4f}"


L.append("## Main table: shares of the RL-tuned model's gain on held-out prompts (95% bootstrap CI)\n")
L.append("| | " + " | ".join(runs) + " |")
L.append("|---|" + "---|" * len(runs))
Rs = {r: read_json(f"{r}.json") for r in runs}
rows = [
    ("base reward", lambda R: f"{R['summary']['base'][R['reward']]:.4f}"),
    ("RL-tuned (LoRA r=32) reward", lambda R: f"{R['summary']['rl_lora'][R['reward']]:.4f}"),
    ("RL gain", lambda R: f"{R['gain']['rl_lora']['gain']:+.4f} [{R['gain']['rl_lora']['ci'][0]:+.4f}, {R['gain']['rl_lora']['ci'][1]:+.4f}]"),
    ("s_mean (mean difference) share %", lambda R: sh(R, "s_mean")),
    ("**s_fit (velocity matching) share %**", lambda R: "**" + sh(R, "s_fit") + "**"),
    ("same-size LoRA (rank 2, out-proj), fitted, share %", lambda R: sh(R, "lora2_fit")),
    ("**s_RL (vectors trained by Flow-GRPO from zero) share %**", lambda R: "**" + sh(R, "s_rl") + "**"),
    ("random vectors, same norms, share %", lambda R: sh(R, "s_rand")),
    ("best CFG scale on the base, share %", lambda R: f"{R['best_cfg_control']['share']:.1f} (CFG {R['best_cfg_control']['cfg']:g})"),
    ("steering parameters", lambda R: f"{R['params']['steering_vectors']:,} ({R['params']['n_vectors']} vectors)"),
    ("RL LoRA parameters", lambda R: f"{R['params']['rl_lora']:,} ({R['params']['lora_over_vectors']:.0f}x)"),
    ("velocity gap closed by s_fit (cond / uncond)", lambda R: f"{R['velocity_gap_closed']['s_fit']['cond']:.2f} / {R['velocity_gap_closed']['s_fit']['uncond']:.2f}"),
    ("velocity gap closed by s_mean (cond / uncond)", lambda R: f"{R['velocity_gap_closed']['s_mean']['cond']:.2f} / {R['velocity_gap_closed']['s_mean']['uncond']:.2f}"),
    ("velocity gap closed by same-size LoRA", lambda R: f"{R['velocity_gap_closed']['lora2']['cond']:.2f} / {R['velocity_gap_closed']['lora2']['uncond']:.2f}"),
]
for name, fn in rows:
    L.append(f"| {name} | " + " | ".join(fn(Rs[r]) for r in runs) + " |")

L.append("\n## Where the fitted vectors act (share % of the RL gain when only these vectors are kept)\n")
L.append("| kept | " + " | ".join(runs) + " |")
L.append("|---|" + "---|" * len(runs))
abl = [k for k in Rs[runs[0]]["share"] if k.startswith("abl_")]
for k in abl + ["kvreset_s_fit", "kvreset_rl_lora"]:
    L.append(f"| {k} | " + " | ".join(sh(Rs[r], k) for r in runs) + " |")

L.append("\n## Side effects: gains on every metric (held-out prompts)\n")
for r in runs:
    R = Rs[r]
    L.append(f"\n**{r}**\n")
    M = ("match", "exact_B", "weight_ok", "legible", "correct_B", "upright", "ink")
    L.append("| policy | match | exact (classifier B + weights) | weight in tercile | p(digit) | both digits right (B) | upright | ink |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append("| base (absolute) | " + " | ".join(f"{R['summary']['base'][m]:.4f}" for m in M) + " |")
    for k, v in R["side_effects"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[m]['gain']:+.4f}" for m in M) + " |")

L.append("\n## CFG sweep on the base (gain on the run's reward)\n")
L.append("| CFG | " + " | ".join(runs) + " |")
L.append("|---|" + "---|" * len(runs))
for w in ("1", "2", "4", "5", "6", "8"):
    L.append(f"| {w} | " + " | ".join(f"{g(Rs[r], 'cfg_' + w)} ({Rs[r]['share']['cfg_' + w]['share']:.0f}%)" for r in runs) + " |")

(RESULTS / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
