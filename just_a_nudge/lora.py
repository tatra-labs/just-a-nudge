"""Minimal LoRA: wrap selected nn.Linear layers of a frozen model."""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * (x @ self.A.t()) @ self.B.t()
        return y


# module-name suffixes inside MMDiT blocks
ATTN_TARGETS = ("img.qkv", "img.out", "txt.qkv", "txt.out")  # Flow-GRPO's SD3 LoRA targets attention
OUT_TARGETS = ("img.out", "txt.out")  # the same-size control: attention output projections only


def inject_lora(model: nn.Module, targets, rank: int, alpha: float | None = None) -> list[LoRALinear]:
    """Freeze `model` and replace matching Linear layers by LoRALinear. Returns the new modules."""
    for p in model.parameters():
        p.requires_grad_(False)
    added = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.startswith("blocks.") and any(full.endswith(t) for t in targets):
                lora = LoRALinear(child, rank, alpha)
                setattr(module, child_name, lora)
                added.append(lora)
    return added


def lora_params(model: nn.Module):
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


def set_lora(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_state(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
