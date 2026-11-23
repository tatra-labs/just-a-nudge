"""Checkpoint and results helpers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import ROOT
from .model import DiTConfig, MMDiT
from .rewards import ClassifierA, ClassifierB

CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def device_default():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_base(model: MMDiT, path=None):
    path = Path(path or CKPT / "base.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg.to_dict(), "state": model.state_dict()}, path)


def load_base(device="cpu", path=None) -> MMDiT:
    ck = torch.load(Path(path or CKPT / "base.pt"), map_location=device, weights_only=True)
    m = MMDiT(DiTConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_classifier(which: str, device="cpu"):
    m = {"A": ClassifierA, "B": ClassifierB}[which]()
    m.load_state_dict(torch.load(CKPT / f"classifier_{which}.pt", map_location=device, weights_only=True))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def write_json(obj, name: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def read_json(name: str):
    return json.loads((RESULTS / name).read_text())


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
