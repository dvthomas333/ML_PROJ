"""Input representations. Each is an nn.Module so its cost is part of the model's measured latency.

spatial  : RGB pixels, per-channel z-score
dct      : log|2-D DCT-II| of the whole 32x32 channel, per-coefficient z-score   (main frequency model)
fft      : log|centred 2-D DFT|, per-coefficient z-score                         (ablation)
blockdct : log|8x8 block DCT| (JPEG-aligned), per-coefficient z-score            (ablation)
"""
import math

import torch
import torch.nn as nn

KINDS = ("spatial", "dct", "fft", "blockdct")


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix: C[k, i] = a_k cos(pi (2i + 1) k / 2n)."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    c = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    c[0] /= math.sqrt(2.0)
    return c.float()


class Representation(nn.Module):
    def __init__(self, kind, size=32, block=8, eps=1e-4, mask=None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.block, self.eps = kind, block, eps
        self.register_buffer("C_full", dct_matrix(size))
        self.register_buffer("C_block", dct_matrix(block))
        stat_shape = (1, 3, 1, 1) if kind == "spatial" else (1, 3, size, size)
        self.register_buffer("mean", torch.zeros(stat_shape))
        self.register_buffer("std", torch.ones(stat_shape))
        self.register_buffer("mask", torch.ones(1, 3, size, size) if mask is None else mask.float().reshape(1, 3, size, size))

    def transform(self, x):
        """x: N x 3 x H x W in [0, 1] -> unnormalised representation of the same shape."""
        if self.kind == "spatial":
            return x
        if self.kind == "dct":
            c = self.C_full @ x @ self.C_full.T
        elif self.kind == "blockdct":
            n, ch, h, w = x.shape
            b = self.block
            blocks = x.reshape(n, ch, h // b, b, w // b, b).permute(0, 1, 2, 4, 3, 5)
            c = (self.C_block @ blocks @ self.C_block.T).permute(0, 1, 2, 4, 3, 5).reshape(n, ch, h, w)
        else:  # fft
            c = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        return torch.log(c.abs() + self.eps)

    def forward(self, x):
        # Spectral transforms are kept in fp32 even under AMP: fp16 loses the small high-frequency coefficients.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = (self.transform(x.float()) - self.mean) / self.std
            return z * self.mask

    @torch.no_grad()
    def fit(self, loader, device):
        """Normalisation statistics from the clean training split only (never re-fitted on perturbed data)."""
        dims = (0, 2, 3) if self.kind == "spatial" else (0,)
        s = s2 = None
        n = 0
        for x, _ in loader:
            z = self.transform(x.to(device).float()).double()
            zs, zs2 = z.sum(dims, keepdim=True), (z * z).sum(dims, keepdim=True)
            s = zs if s is None else s + zs
            s2 = zs2 if s2 is None else s2 + zs2
            n += z.shape[0] * (z.shape[2] * z.shape[3] if self.kind == "spatial" else 1)
        mean = s / n
        std = (s2 / n - mean**2).clamp_min(1e-12).sqrt()
        self.mean.copy_(mean.float())
        self.std.copy_(std.float())
