"""Matched lightweight CNNs. All models share one backbone; only the input representation differs.

Single-domain (width 32): 3 x [Conv3x3-BN-ReLU-Conv3x3-BN-ReLU-MaxPool], 32-64-128 ch, GAP, Dropout, Linear -> 1 logit
Hybrid (width 23 per branch): spatial branch + DCT branch, GAP features concatenated, Dropout, Linear -> 1 logit
"""
import time

import torch
import torch.nn as nn

from .representations import Representation

MODEL_NAMES = ("spatial", "dct", "fft", "blockdct", "hybrid")


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Backbone(nn.Module):
    def __init__(self, width=32, in_ch=3):
        super().__init__()
        w = (width, 2 * width, 4 * width)
        self.blocks = nn.Sequential(conv_block(in_ch, w[0]), conv_block(w[0], w[1]), conv_block(w[1], w[2]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = w[2]

    def forward(self, x):
        return self.pool(self.blocks(x)).flatten(1)

    @property
    def gradcam_layer(self):
        return self.blocks[-1][5]  # last ReLU before pooling, 8x8 map for 32x32 input


class SingleDomainCNN(nn.Module):
    def __init__(self, kind, width=32, dropout=0.3, **rep_kw):
        super().__init__()
        self.rep = Representation(kind, **rep_kw)
        self.backbone = Backbone(width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_dim, 1))

    def representations(self):
        return [self.rep]

    def forward_from_rep(self, z):
        return self.head(self.backbone(z)).squeeze(1)

    def forward(self, x):
        return self.forward_from_rep(self.rep(x))


class HybridCNN(nn.Module):
    def __init__(self, width=23, dropout=0.3, **rep_kw):
        super().__init__()
        rep_kw.pop("mask", None)
        self.rep_s = Representation("spatial", **rep_kw)
        self.rep_f = Representation("dct", **rep_kw)
        self.bb_s, self.bb_f = Backbone(width), Backbone(width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * self.bb_s.out_dim, 1))

    def representations(self):
        return [self.rep_s, self.rep_f]

    def forward(self, x):
        h = torch.cat([self.bb_s(self.rep_s(x)), self.bb_f(self.rep_f(x))], dim=1)
        return self.head(h).squeeze(1)


def build_model(name, cfg, mask=None, dropout=None):
    m, r = cfg["model"], cfg["representation"]
    drop = m["dropout"] if dropout is None else dropout
    rep_kw = {"eps": r["log_eps"], "block": r["block_size"], "mask": mask}
    if name == "hybrid":
        return HybridCNN(m["hybrid_width"], drop, **rep_kw)
    if name not in MODEL_NAMES:
        raise ValueError(name)
    return SingleDomainCNN(name, m["width"], drop, **rep_kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_macs(model, size=32):
    """Multiply-accumulates of Conv2d and Linear layers for one image (transform cost excluded)."""
    macs = 0

    def conv_hook(mod, _inp, out):
        nonlocal macs
        kh, kw = mod.kernel_size
        macs += out[0].numel() * (mod.in_channels // mod.groups) * kh * kw

    def lin_hook(mod, _inp, _out):
        nonlocal macs
        macs += mod.in_features * mod.out_features

    hooks = [
        mod.register_forward_hook(conv_hook if isinstance(mod, nn.Conv2d) else lin_hook)
        for mod in model.modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    model(torch.rand(1, 3, size, size, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


@torch.no_grad()
def cpu_latency_ms(model, threads=1, warmup=50, runs=500, size=32):
    """Single-image CPU latency including the domain transform. Returns (median, p95) in ms."""
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.rand(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        t.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(t)
    return float(t.median()), float(t.quantile(0.95))
