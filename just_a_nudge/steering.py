"""Steering vectors, exactly in the paper's layout (Eq. 1).

One vector per (block output, stream, guidance pass, step bin), added to every token of that
stream at that block output:  h~ = h + alpha * s[stream, block, pass, bin(t)].
Passes: 0 = conditional, 1 = unconditional. Bins: 0 = early (t >= tau, high noise), 1 = late.
"""
from __future__ import annotations

import torch
import torch.nn as nn

COND, UNCOND = 0, 1
EARLY, LATE = 0, 1


class SteeringVectors(nn.Module):
    def __init__(self, n_img: int, n_txt: int, width: int, n_pass: int = 2, n_bins: int = 2, alpha: float = 1.0):
        super().__init__()
        self.img = nn.Parameter(torch.zeros(n_pass, n_bins, n_img, width))
        self.txt = nn.Parameter(torch.zeros(n_pass, n_bins, n_txt, width))
        self.alpha = alpha
        # ablation masks (1 = keep); same shapes as the vectors
        self.register_buffer("img_mask", torch.ones(n_pass, n_bins, n_img, 1))
        self.register_buffer("txt_mask", torch.ones(n_pass, n_bins, n_txt, 1))

    @classmethod
    def for_model(cls, model, **kw):
        return cls(model.n_img_outputs, model.n_txt_outputs, model.cfg.width, **kw)

    def n_vectors(self) -> int:
        return self.img.shape[0] * self.img.shape[1] * (self.img.shape[2] + self.txt.shape[2])

    def numel(self) -> int:
        return self.img.numel() + self.txt.numel()

    def get(self, pass_idx: torch.Tensor, bin_idx: torch.Tensor):
        """Per-sample vectors: (B, n_img, D) and (B, n_txt, D)."""
        img = (self.img * self.img_mask)[pass_idx, bin_idx] * self.alpha
        txt = (self.txt * self.txt_mask)[pass_idx, bin_idx] * self.alpha
        return img, txt

    # ---- ablations -------------------------------------------------------------------------
    def reset_mask(self):
        self.img_mask.fill_(1.0)
        self.txt_mask.fill_(1.0)

    def keep_only(self, *, passes=None, bins=None, streams=None, blocks=None):
        """Zero every vector outside the selection (None = keep all along that axis)."""
        self.reset_mask()
        P, K, Li, _ = self.img_mask.shape
        Lt = self.txt_mask.shape[2]
        for name, mask, L in (("img", self.img_mask, Li), ("txt", self.txt_mask, Lt)):
            if streams is not None and name not in streams:
                mask.zero_()
                continue
            for p in range(P):
                for k in range(K):
                    for l in range(L):
                        keep = (passes is None or p in passes) and (bins is None or k in bins) and (
                            blocks is None or l in blocks
                        )
                        if not keep:
                            mask[p, k, l] = 0.0

    def mask_view(self):
        """Effective (masked) vectors, for saving or norm statistics."""
        return self.img * self.img_mask, self.txt * self.txt_mask
