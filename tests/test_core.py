import copy
import math

import numpy as np
import pytest
import torch

from just_a_nudge.evaluate import EvalSet, make_vfn, per_prompt, sample_with, share_stats, gain_stats
from just_a_nudge.fit import mean_difference, random_like, velocity_gap, velocity_match
from just_a_nudge.lora import ATTN_TARGETS, OUT_TARGETS, inject_lora, lora_params, set_lora
from just_a_nudge.model import DiTConfig, MMDiT, n_params, null_tokens, prompt_tokens
from just_a_nudge.rewards import hard_weight, soft_weight, slant, upright
from just_a_nudge.sampling import gauss_logp, model_velocity, sde_mean_std, timesteps
from just_a_nudge.steering import SteeringVectors

TINY = DiTConfig(width=32, depth=3, heads=2, mlp_ratio=2)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = MMDiT(TINY)
    # adaLN-Zero starts at identity; randomise so every path is exercised
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 56, generator=g)
    t = torch.rand(n, generator=g)
    pr = torch.stack([torch.randint(hi, (n,), generator=g) for hi in (10, 3, 10, 3)], 1)
    return x, t, prompt_tokens(pr)


def test_shapes_and_counts():
    m = tiny_model()
    x, t, tok = batch()
    assert m(x, t, tok).shape == x.shape
    s = SteeringVectors.for_model(m)
    L, D = TINY.depth, TINY.width
    assert s.n_vectors() == (2 * L - 1) * 2 * 2  # (img L + txt L-1) x passes x bins
    assert s.numel() == (2 * L - 1) * D * 2 * 2


def test_zero_vectors_are_identity():
    m = tiny_model()
    x, t, tok = batch()
    s = SteeringVectors.for_model(m)
    p = torch.zeros(len(x), dtype=torch.long)
    assert torch.equal(model_velocity(m, x, t, tok, p, s), m(x, t, tok))


def test_vectors_add_to_every_token_of_each_block_output():
    m = tiny_model()
    x, t, tok = batch()
    s = SteeringVectors.for_model(m)
    with torch.no_grad():
        s.img[0, :, 0] = 1.5
        s.txt[0, :, 0] = -0.5
    r0, r1 = {}, {}
    m(x, t, tok, record=r0)
    p = torch.zeros(len(x), dtype=torch.long)
    model_velocity(m, x, t, tok, p, s, record=r1)
    assert torch.allclose(r1["img_out"][0] - r0["img_out"][0], torch.full_like(r0["img_out"][0], 1.5), atol=1e-5)
    assert torch.allclose(r1["txt_out"][0] - r0["txt_out"][0], torch.full_like(r0["txt_out"][0], -0.5), atol=1e-5)
    # the unconditional pass uses its own vectors (still zero here)
    r2 = {}
    model_velocity(m, x, t, tok, torch.ones(len(x), dtype=torch.long), s, record=r2)
    assert torch.allclose(r2["img_out"][0], r0["img_out"][0])


def test_step_bins_select_vectors():
    m = tiny_model()
    x, _, tok = batch()
    s = SteeringVectors.for_model(m)
    with torch.no_grad():
        s.img[0, 1] = 3.0  # late bin only
    p = torch.zeros(len(x), dtype=torch.long)
    early = torch.full((len(x),), 0.9)
    late = torch.full((len(x),), 0.1)
    assert torch.equal(model_velocity(m, x, early, tok, p, s), m(x, early, tok))
    assert not torch.allclose(model_velocity(m, x, late, tok, p, s), m(x, late, tok))


def test_keep_only_masks():
    s = SteeringVectors(3, 2, 8)
    with torch.no_grad():
        s.img.fill_(1.0)
        s.txt.fill_(1.0)
    s.keep_only(passes=[0], streams=["txt"])
    img, txt = s.mask_view()
    assert img.abs().sum() == 0
    assert txt[0].abs().sum() > 0 and txt[1].abs().sum() == 0
    s.keep_only(bins=[1], blocks=[0])
    img, txt = s.mask_view()
    assert img[:, 0].abs().sum() == 0 and img[:, 1, 1:].abs().sum() == 0 and img[:, 1, 0].abs().sum() > 0


def test_lora_zero_init_toggle_and_same_size_control():
    m = tiny_model()
    x, t, tok = batch()
    ref = m(x, t, tok)
    m2 = copy.deepcopy(m)
    added = inject_lora(m2, ATTN_TARGETS, rank=4)
    assert len(added) == 2 * TINY.depth - 1 + 2 * TINY.depth - 1 + 1  # qkv on all 2L-1+1 streams, out on 2L-1
    assert torch.allclose(m2(x, t, tok), ref, atol=1e-6)
    with torch.no_grad():
        for p in lora_params(m2):
            p.add_(0.1)
    assert not torch.allclose(m2(x, t, tok), ref, atol=1e-4)
    set_lora(m2, False)
    assert torch.allclose(m2(x, t, tok), ref, atol=1e-6)
    m3 = copy.deepcopy(m)
    inject_lora(m3, OUT_TARGETS, rank=2)
    assert sum(p.numel() for p in lora_params(m3)) == SteeringVectors.for_model(m).numel()


def test_sde_step_reduces_to_euler_and_logp():
    x = torch.randn(2, 1, 28, 56)
    v = torch.randn_like(x)
    mean, s = sde_mean_std(x, v, 0.6, 0.5, 0.9, eta=0.0)
    assert s == 0.0
    assert torch.allclose(mean, x + (0.5 - 0.6) * v)
    mean, s = sde_mean_std(x, v, 0.6, 0.5, 0.9, eta=0.7)
    assert s > 0
    lp = gauss_logp(mean, mean, s)
    assert torch.allclose(lp, torch.full((2,), -math.log(s) - 0.5 * math.log(2 * math.pi)))
    ts = timesteps(10)
    assert ts[0] == 1.0 and ts[-1] == 0.0 and len(ts) == 11


def _fit_data(m, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x0": torch.randn(n, 1, 28, 56, generator=g).clamp(-1, 1),
        "prompt": torch.stack([torch.randint(hi, (n,), generator=g) for hi in (10, 3, 10, 3)], 1),
    }


def test_velocity_matching_recovers_a_constant_shift():
    base = tiny_model()
    for p in base.parameters():
        p.requires_grad_(False)
    true = SteeringVectors.for_model(base)
    with torch.no_grad():
        true.img.normal_(0, 0.3)
        true.txt.normal_(0, 0.3)
    teacher = lambda x, t, tok, p: model_velocity(base, x, t, tok, p, true)  # noqa: E731
    fit = SteeringVectors.for_model(base)
    student = lambda x, t, tok, p: model_velocity(base, x, t, tok, p, fit)  # noqa: E731
    data = _fit_data(base)
    velocity_match(student, [fit.img, fit.txt], teacher, data, updates=150, batch=16, lr=5e-2, log=lambda *_: None)
    closed = velocity_gap(student, teacher, lambda x, t, tok, p: base(x, t, tok), data, n=64)
    assert closed["cond"] > 0.97 and closed["uncond"] > 0.97


def test_mean_difference_is_exact_for_a_constant_shift():
    base = tiny_model()
    true = SteeringVectors.for_model(base)
    with torch.no_grad():
        true.img.normal_(0, 0.3)
        true.txt.normal_(0, 0.3)
    n, S = 8, 4
    g = torch.Generator().manual_seed(1)
    data = {
        "traj": torch.randn(n, S, 1, 28, 56, generator=g),
        "traj_prompt": torch.stack([torch.randint(hi, (n,), generator=g) for hi in (10, 3, 10, 3)], 1),
        "ts": [1.0, 0.75, 0.5, 0.25, 0.0],  # two levels per bin
    }
    teacher = lambda x, t, tok, p, rec: model_velocity(base, x, t, tok, p, true, record=rec)  # noqa: E731
    est = mean_difference(base, SteeringVectors.for_model(base), teacher, data, batch=4)
    assert torch.allclose(est.img, true.img, atol=1e-4)
    assert torch.allclose(est.txt, true.txt, atol=1e-4)
    rnd = random_like(est, SteeringVectors.for_model(base))
    assert torch.allclose(rnd.img.norm(dim=-1), est.img.norm(dim=-1), atol=1e-5)


def test_kv_reset_is_identity_without_changes_and_differs_with_vectors():
    m = tiny_model()
    es = EvalSet([(1, 0, 2, 2), (3, 1, 4, 0)], per_prompt=2, device="cpu")
    tok = prompt_tokens(es.prompt)
    a = sample_with(make_vfn(m, w=2.0), tok, es.noise, n_steps=3)
    b = sample_with(make_vfn(m, w=2.0, kv_reset=True), tok, es.noise, n_steps=3)
    assert torch.allclose(a, b, atol=1e-5)
    s = SteeringVectors.for_model(m)
    with torch.no_grad():
        s.txt.normal_(0, 0.5)  # (a constant fill would be invisible: every reader is a LayerNorm)
    c = sample_with(make_vfn(m, s, w=2.0), tok, es.noise, n_steps=3)
    d = sample_with(make_vfn(m, s, w=2.0, kv_reset=True), tok, es.noise, n_steps=3)
    assert not torch.allclose(c, a, atol=1e-4)
    assert not torch.allclose(c, d, atol=1e-4)


def test_share_and_gain_stats():
    pid = np.repeat(np.arange(10), 4)
    base = np.zeros(40)
    rl = np.ones(40) + 0.1 * np.random.default_rng(0).standard_normal(40)
    half = 0.5 * rl
    st = share_stats(base, rl, half, pid)
    assert abs(st["share"] - 50.0) < 1e-9 and st["ci"][0] <= 50.0 <= st["ci"][1]
    gs = gain_stats(base, rl, pid)
    assert gs["ci"][0] < gs["gain"] < gs["ci"][1]
    assert per_prompt(np.arange(40.0), pid)[0] == pytest.approx(1.5)


def test_upright_reward():
    img = torch.zeros(3, 1, 28, 56) - 1  # black background in [-1, 1]
    img[0, 0, 4:24, 13:15] = 1  # vertical bars in both halves
    img[0, 0, 4:24, 41:43] = 1
    for r in range(4, 24):  # slanted bars
        c = 10 + (24 - r) // 3
        img[1, 0, r, c : c + 2] = 1
        img[1, 0, r, 28 + c : 30 + c] = 1
    s_vert, _ = slant(((img[0:1, :, :, :28] + 1) / 2))
    assert abs(s_vert.item()) < 1e-6
    r = upright(img)
    assert r[0] > r[1]  # upright beats slanted
    assert r[2] < r[0]  # blank image is penalised for missing ink
    assert n_params(MMDiT(TINY)) > 0
    assert null_tokens(2, "cpu").shape == (2, 6)


def test_prompt_tokens_and_weight_checks():
    pr = torch.tensor([[3, 2, 7, 0]])
    assert prompt_tokens(pr).tolist() == [[11, 3, 15, 7, 13, 12]]
    th = torch.tensor([[50.0, 80.0]] * 10)
    ink = torch.tensor([40.0, 65.0, 95.0])
    cls = torch.tensor([0, 0, 0])
    assert hard_weight(ink, cls, th).tolist() == [0, 1, 2]
    for w in range(3):
        sw = soft_weight(ink, cls, torch.full((3,), w), th)
        assert sw.argmax().item() == w and sw.max() > 0.95
    total = sum(soft_weight(ink, cls, torch.full((3,), w), th) for w in range(3))
    assert torch.allclose(total, torch.ones(3))
