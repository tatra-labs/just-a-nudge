"""Stage 3: figures from results/*.json and results/*_samples.pt."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from just_a_nudge.data import prompt_text  # noqa: E402
from just_a_nudge.io import FIGURES, RESULTS, read_json  # noqa: E402

FIGURES.mkdir(exist_ok=True)
XMIN = -80  # bars below this are drawn clipped and labelled
REWARDS = [r for r in ("match", "upright", "match_weakbase") if (RESULTS / f"{r}.json").exists()]
LABEL = {"match": "match (GenEval/OCR analogue)", "upright": "upright (PickScore analogue)",
         "match_weakbase": "match, weak (4.5k-step) base"}

COLS = [("base", "base"), ("rl_lora", "RL-tuned (LoRA)"), ("s_fit", "base + fitted vectors"),
        ("s_rl", "base + RL-trained vectors"), ("s_mean", "base + mean-diff vectors"), ("cfg_8", "base, CFG 8")]


def grid(reward, rows=8):
    S = torch.load(RESULTS / f"{reward}_samples.pt", weights_only=False)
    cols = [(k, t) for k, t in COLS if k in S]
    # rows: evenly spaced held-out prompts (they are sorted by digits, so this spans 0-9)
    n = len(S["test_prompts"])
    idx = [int(round(j * (n - 1) / (rows - 1))) for j in range(rows)]
    fig, axes = plt.subplots(rows, len(cols), figsize=(2.0 * len(cols) + 0.6, 1.05 * rows))
    for j, (k, title) in enumerate(cols):
        imgs = S[k]
        for i in range(rows):
            ax = axes[i, j]
            ax.imshow(((imgs[idx[i], 0].float() + 1) / 2).clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                label = prompt_text(S["test_prompts"][idx[i]]).replace(", ", chr(10))
                ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=24, va="center")
    fig.suptitle(f"{LABEL[reward]}: same prompt and starting noise in every row", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{reward}_grid.png", dpi=130)
    plt.close(fig)


def shares():
    keys = [("s_mean", "mean-difference vectors"), ("s_fit", "fitted vectors (velocity matching)"),
            ("lora2_fit", "same-size LoRA, fitted"), ("s_rl", "vectors trained by RL from zero"),
            ("s_rand", "random vectors, same norms")]
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(5.2 * len(REWARDS), 2.8), squeeze=False)
    for ax, reward in zip(axes[0], REWARDS):
        R = read_json(f"{reward}.json")
        ks = [(k, t) for k, t in keys if k in R["share"]]
        bc = R["best_cfg_control"]
        ks.append((f"cfg_{bc['cfg']:g}", f"best CFG scale ({bc['cfg']:g})"))
        vals = [R["share"][k]["share"] for k, _ in ks]
        lo = [R["share"][k]["share"] - R["share"][k]["ci"][0] for k, _ in ks]
        hi = [R["share"][k]["ci"][1] - R["share"][k]["share"] for k, _ in ks]
        y = np.arange(len(ks))[::-1]
        clipped = [max(v, XMIN) for v in vals]
        err = [[l if v > XMIN else 0 for l, v in zip(lo, vals)], [h if v > XMIN else 0 for h, v in zip(hi, vals)]]
        ax.barh(y, clipped, xerr=err, color=["#d07a4a" if v < 0 else "#4a7bd0" for v in vals], capsize=3)
        for yy, v in zip(y, vals):
            if v <= XMIN:
                ax.text(XMIN + 3, yy, f"{v:.0f}% (off scale)", va="center", fontsize=7, color="white")
        ax.set_xlim(XMIN, 130)
        ax.axvline(100, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([t for _, t in ks], fontsize=8)
        ax.set_xlabel("share of the RL-tuned model's reward gain (%)", fontsize=8)
        ax.set_title(LABEL[reward], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "shares.png", dpi=140)
    plt.close(fig)


def ablations():
    names = [("abl_cond_only", "conditional pass only"), ("abl_uncond_only", "unconditional pass only"),
             ("abl_early_only", "early bin only"), ("abl_late_only", "late bin only"),
             ("abl_text_only", "text stream only"), ("abl_image_only", "image stream only"),
             ("abl_cond_text_early", "cond + text + early"), ("kvreset_s_fit", "vectors, text K/V reset"),
             ("kvreset_rl_lora", "RL model, text K/V reset")]
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(5.2 * len(REWARDS), 3.2), squeeze=False)
    for ax, reward in zip(axes[0], REWARDS):
        R = read_json(f"{reward}.json")
        ks = [(k, t) for k, t in names if k in R["share"]]
        vals = [R["share"][k]["share"] for k, _ in ks]
        y = np.arange(len(ks))[::-1]
        ax.barh(y, [max(v, XMIN) for v in vals], color=["#d07a4a" if v < 0 else "#4a7bd0" for v in vals])
        for yy, v in zip(y, vals):
            if v <= XMIN:
                ax.text(XMIN + 3, yy, f"{v:.0f}% (off scale)", va="center", fontsize=7, color="white")
        ax.set_xlim(XMIN, 130)
        ax.axvline(100, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([t for _, t in ks], fontsize=8)
        ax.set_xlabel("share of the RL-tuned model's gain (%)", fontsize=8)
        ax.set_title(f"{LABEL[reward]}: where the fitted vectors act", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablations.png", dpi=140)
    plt.close(fig)


def curves():
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(5.2 * len(REWARDS), 2.6), squeeze=False)
    for ax, reward in zip(axes[0], REWARDS):
        R = read_json(f"{reward}.json")
        for k, lab in (("rl_lora", "LoRA (rank 32)"), ("s_rl", "steering vectors only")):
            h = R["history"].get(k)
            if not h:
                continue
            r = np.array([e["reward"] for e in h])
            k_ = 10
            sm = np.convolve(r, np.ones(k_) / k_, mode="valid")
            ax.plot(np.arange(len(sm)) + k_ - 1, sm, label=lab)
        ax.set_xlabel("Flow-GRPO update", fontsize=8)
        ax.set_ylabel("training reward (10-update mean)", fontsize=8)
        ax.set_title(LABEL[reward], fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "rl_curves.png", dpi=140)
    plt.close(fig)


for r in REWARDS:
    grid(r)
shares()
ablations()
curves()
print("figures written to", FIGURES)
