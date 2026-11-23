"""A tiny MM-DiT (SD3-style) for two-digit "text-to-image" generation.

Two streams, as in Stable Diffusion 3: a text stream (6 prompt tokens) and an image stream
(98 patches of a 28x56 image). Each block modulates both streams with the timestep (adaLN-Zero),
runs one *joint* attention over the concatenated tokens, and keeps separate projections and MLPs
per stream. The last block is "context pre-only": its text stream only supplies keys and values,
so there are L image-stream block outputs and L-1 text-stream block outputs, which is where the
steering vectors go (SD3.5-M: 24 + 23 = 47).

The forward pass can
  * add steering vectors to every block output (`steer=(img_vecs, txt_vecs)`),
  * record block outputs and the text keys/values each block exposes to image queries, and
  * replace the text keys/values read by image queries (the paper's Sec. 5 attention reset).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_H, IMG_W, PATCH = 28, 56, 4
N_IMG_TOK = (IMG_H // PATCH) * (IMG_W // PATCH)  # 7 * 14 = 98
NULL, BOS, EOS = 10, 11, 12
THIN, REGULAR, BOLD = 13, 14, 15  # stroke-weight words; style s in {0,1,2} is token 13 + s
VOCAB = 16
TXT_LEN = 6  # [BOS, left digit, left weight, right digit, right weight, EOS]


@dataclass
class DiTConfig:
    width: int = 128
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4

    def to_dict(self):
        return asdict(self)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Stream(nn.Module):
    """Per-stream weights of one MM-DiT block."""

    def __init__(self, d: int, heads: int, mlp_ratio: int, pre_only: bool = False):
        super().__init__()
        self.pre_only = pre_only
        self.heads = heads
        self.ada = nn.Linear(d, (2 if pre_only else 6) * d)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.q_norm = RMSNorm(d // heads)
        self.k_norm = RMSNorm(d // heads)
        if not pre_only:
            self.out = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.fc1 = nn.Linear(d, mlp_ratio * d)
            self.fc2 = nn.Linear(mlp_ratio * d, d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def qkv_heads(self, x, c):
        mods = self.ada(F.silu(c)).chunk(2 if self.pre_only else 6, dim=-1)
        h = modulate(self.norm1(x), mods[0], mods[1])
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        return self.q_norm(q), self.k_norm(k), v, mods

    def finish(self, x, attn_out, mods):
        _, _, g1, s2, sc2, g2 = mods
        x = x + g1[:, None] * self.out(attn_out)
        h = modulate(self.norm2(x), s2, sc2)
        return x + g2[:, None] * self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


def _merge(o):  # (B, H, T, hd) -> (B, T, D)
    B, H, T, hd = o.shape
    return o.transpose(1, 2).reshape(B, T, H * hd)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio, last=False):
        super().__init__()
        self.img = Stream(d, heads, mlp_ratio)
        self.txt = Stream(d, heads, mlp_ratio, pre_only=last)
        self.last = last

    def forward(self, img, txt, c, txt_kv_for_img=None, record=None):
        qi, ki, vi, mi = self.img.qkv_heads(img, c)
        qt, kt, vt, mt = self.txt.qkv_heads(txt, c)
        if record is not None:
            record.setdefault("txt_kv", []).append((kt, vt))
        k = torch.cat([kt, ki], dim=2)
        v = torch.cat([vt, vi], dim=2)
        T = txt.shape[1]
        if txt_kv_for_img is None:
            q = torch.cat([qt, qi], dim=2)
            o = F.scaled_dot_product_attention(q, k, v)
            ot, oi = o[:, :, :T], o[:, :, T:]
        else:  # image queries read replaced text keys/values; text queries read the real ones
            kt2, vt2 = txt_kv_for_img
            oi = F.scaled_dot_product_attention(qi, torch.cat([kt2, ki], 2), torch.cat([vt2, vi], 2))
            ot = F.scaled_dot_product_attention(qt, k, v)
        img = self.img.finish(img, _merge(oi), mi)
        txt = None if self.last else self.txt.finish(txt, _merge(ot), mt)
        return img, txt


class MMDiT(nn.Module):
    def __init__(self, cfg: DiTConfig = DiTConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.patch_in = nn.Linear(PATCH * PATCH, d)
        self.img_pos = nn.Parameter(torch.randn(1, N_IMG_TOK, d) * 0.02)
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.txt_pos = nn.Parameter(torch.randn(1, TXT_LEN, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [Block(d, cfg.heads, cfg.mlp_ratio, last=(i == cfg.depth - 1)) for i in range(cfg.depth)]
        )
        self.final_ada = nn.Linear(d, 2 * d)
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.patch_out = nn.Linear(d, PATCH * PATCH)
        for m in (self.final_ada, self.patch_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def n_img_outputs(self):
        return self.cfg.depth

    @property
    def n_txt_outputs(self):
        return self.cfg.depth - 1

    def patchify(self, x):  # (B,1,28,56) -> (B,98,16)
        B = x.shape[0]
        p = PATCH
        x = x.view(B, IMG_H // p, p, IMG_W // p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(B, N_IMG_TOK, p * p)

    def unpatchify(self, x):
        B = x.shape[0]
        p = PATCH
        x = x.view(B, IMG_H // p, IMG_W // p, p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(B, 1, IMG_H, IMG_W)

    def forward(self, x, t, tokens, steer=None, record=None, txt_kv_for_img=None):
        """x: (B,1,28,56) noisy image; t: (B,) in [0,1] (1 = noise); tokens: (B,4) long.

        steer: optional (img_vecs (B,L,D), txt_vecs (B,L-1,D)) added to every token of each block output.
        record: optional dict; filled with 'img_out', 'txt_out' (lists of (B,T,D)) and 'txt_kv'.
        txt_kv_for_img: optional list (per block) of (k, v) text tensors that image queries read instead.
        """
        img = self.patch_in(self.patchify(x)) + self.img_pos
        txt = self.tok_emb(tokens) + self.txt_pos
        c = self.t_mlp(timestep_embedding(t, 256))
        for i, blk in enumerate(self.blocks):
            kv = None if txt_kv_for_img is None else txt_kv_for_img[i]
            img, txt = blk(img, txt, c, txt_kv_for_img=kv, record=record)
            if steer is not None:
                img = img + steer[0][:, i][:, None]
                if txt is not None:
                    txt = txt + steer[1][:, i][:, None]
            if record is not None:
                record.setdefault("img_out", []).append(img)
                if txt is not None:
                    record.setdefault("txt_out", []).append(txt)
        sh, sc = self.final_ada(F.silu(c)).chunk(2, dim=-1)
        out = self.patch_out(modulate(self.final_norm(img), sh, sc))
        return self.unpatchify(out)


def prompt_tokens(pr: torch.Tensor) -> torch.Tensor:
    """pr: (B, 4) long = (left digit, left weight 0-2, right digit, right weight 0-2) -> (B, 6) tokens."""
    B = pr.shape[0]
    tok = torch.empty(B, TXT_LEN, dtype=torch.long, device=pr.device)
    tok[:, 0] = BOS
    tok[:, 1] = pr[:, 0]
    tok[:, 2] = THIN + pr[:, 1]
    tok[:, 3] = pr[:, 2]
    tok[:, 4] = THIN + pr[:, 3]
    tok[:, 5] = EOS
    return tok


def null_tokens(B: int, device) -> torch.Tensor:
    tok = torch.full((B, TXT_LEN), NULL, dtype=torch.long, device=device)
    tok[:, 0] = BOS
    tok[:, -1] = EOS
    return tok


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
