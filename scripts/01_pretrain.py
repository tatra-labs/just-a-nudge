"""Stage 1: train the two MNIST classifiers (reward A, eval-only B) and pretrain the base MM-DiT."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from just_a_nudge.data import load_mnist  # noqa: E402
from just_a_nudge.io import CKPT, Logger, device_default, save_base, write_json  # noqa: E402
from just_a_nudge.model import n_params  # noqa: E402
from just_a_nudge.pretrain import train_base  # noqa: E402
from just_a_nudge.rewards import ClassifierA, ClassifierB, accuracy, train_classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000)  # 20000 saturates the rewards; see README
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--skip-classifiers", action="store_true")
ap.add_argument("--snapshots", default="", help="comma-separated steps at which to save EMA snapshots")
ap.add_argument("--out", default="base.pt", help="checkpoint file name under checkpoints/")
args = ap.parse_args()

dev = device_default()
log = Logger(Path(__file__).resolve().parent.parent / "logs" / f"01_pretrain_{Path(args.out).stem}.log")
CKPT.mkdir(exist_ok=True)
info = {}
if not args.skip_classifiers:
    xtr, ytr = load_mnist(True)
    xte, yte = load_mnist(False)
    for name, cls, seed in (("A", ClassifierA, 1), ("B", ClassifierB, 2)):
        clf = train_classifier(cls(), xtr, ytr, epochs=3, device=dev, seed=seed, log=log)
        acc = accuracy(clf, xte, yte, device=dev)
        log(f"classifier {name}: MNIST test accuracy {acc:.4f}")
        torch.save(clf.state_dict(), CKPT / f"classifier_{name}.pt")
        info[f"classifier_{name}_mnist_test_acc"] = acc

from just_a_nudge.data import weight_thresholds  # noqa: E402

xtr, ytr = load_mnist(True)
torch.save(weight_thresholds(xtr, ytr), CKPT / "weight_thresholds.pt")
snaps = tuple(int(x) for x in args.snapshots.split(",") if x)
ema = train_base(steps=args.steps, batch=args.batch, device=dev, log=log, snapshots=snaps,
                 on_snapshot=lambda step, m: save_base(m, CKPT / f"base_step{step}.pt"))
save_base(ema, CKPT / args.out)
info.update({"base_params": n_params(ema), "base_steps": args.steps, "base_batch": args.batch})
log(str(info))
write_json(info, f"pretrain_{Path(args.out).stem}.json")
