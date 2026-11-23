"""Base model sanity check: rewards on the held-out prompts across CFG scales (4 seeds per prompt)."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from just_a_nudge.data import prompt_split  # noqa: E402
from just_a_nudge.evaluate import EvalSet, evaluate, make_vfn, summarize  # noqa: E402
from just_a_nudge.io import CKPT, load_base, load_classifier, write_json  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--cfgs", default="1,1.5,2,3,4.5,6")
args = ap.parse_args()
base = load_base(dev, args.ckpt)
A, B = load_classifier("A", dev), load_classifier("B", dev)
_, test = prompt_split()
es = EvalSet(test, per_prompt=4, device=dev)
th = torch.load(CKPT / "weight_thresholds.pt", weights_only=True)
out = {}
for w in [float(c) for c in args.cfgs.split(",")]:
    out[w] = {k: round(v, 4) for k, v in summarize(evaluate(make_vfn(base, w=w), es, A, B, th)).items()}
    print(w, out[w], flush=True)
name = Path(args.ckpt).stem if args.ckpt else "base"
write_json({str(k): v for k, v in out.items()}, f"cfg_check_{name}.json")
