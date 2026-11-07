"""Deterministic post-processing perturbations, applied in the pixel domain to uint8 H x W x 3 images.

Every perturbation is applied *before* the domain transform, so the frequency models see the spectrum of
the perturbed image, exactly as they would in deployment.
"""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

JPEG_SUBSAMPLING = 2  # Pillow code for 4:2:0 chroma subsampling (the common default for photographs)
REENCODE_Q = (70, 95)  # per-cycle JPEG quality range of the re-encoding chain


def jpeg(img, quality):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING, optimize=False)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB")).copy()


def resize(img, scale):
    """Down-scale by `scale` then restore the original size (bilinear; Pillow anti-aliases the down-scale)."""
    h, w = img.shape[:2]
    small = (max(1, round(w * scale)), max(1, round(h * scale)))
    im = Image.fromarray(img).resize(small, Image.Resampling.BILINEAR)
    return np.asarray(im.resize((w, h), Image.Resampling.BILINEAR))


def gaussian_blur(img, sigma):
    """Exact separable Gaussian, kernel radius ceil(3*sigma), half-sample symmetric padding."""
    if sigma <= 0:
        return img.copy()
    r = int(np.ceil(3 * sigma))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(t**2) / (2 * sigma**2))
    k /= k.sum()
    out = convolve1d(img.astype(np.float64), k, axis=0, mode="reflect")
    out = convolve1d(out, k, axis=1, mode="reflect")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def reencode(img, cycles, rng):
    """Repeated JPEG re-encoding with a fresh quality in REENCODE_Q at each cycle (sharing across platforms)."""
    for _ in range(int(cycles)):
        img = jpeg(img, rng.integers(REENCODE_Q[0], REENCODE_Q[1] + 1))
    return img


def compound(img):
    """A realistic chain: resize 0.75 -> blur sigma 0.5 -> JPEG Q75."""
    return jpeg(gaussian_blur(resize(img, 0.75), 0.5), 75)


@dataclass(frozen=True)
class Condition:
    family: str
    level: float

    @property
    def name(self):
        return "clean" if self.family == "clean" else f"{self.family}_{self.level:g}"

    def apply(self, img, rng):
        if self.family == "clean":
            return img.copy()
        if self.family == "jpeg":
            return jpeg(img, self.level)
        if self.family == "resize":
            return resize(img, self.level)
        if self.family == "blur":
            return gaussian_blur(img, self.level)
        if self.family == "reencode":
            return reencode(img, self.level, rng)
        if self.family == "compound":
            return compound(img)
        raise ValueError(self.family)


def condition_grid():
    """The 27 test conditions (1 clean + 26 perturbed)."""
    grid = [Condition("clean", 0)]
    grid += [Condition("jpeg", q) for q in (95, 90, 80, 70, 60, 50, 40, 30, 20, 10)]
    grid += [Condition("resize", s) for s in (0.9, 0.75, 0.5, 0.375, 0.25)]
    grid += [Condition("blur", s) for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    grid += [Condition("reencode", k) for k in (2, 5, 10, 20)]
    grid += [Condition("compound", 1)]
    return grid


def apply_condition(x, cond, seed):
    """Apply one condition to an N x H x W x 3 array; image i always gets rng seeded by (seed, i)."""
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = cond.apply(x[i], np.random.default_rng([seed, i]))
    return out


class RandomPerturbation:
    """Training-time augmentation for regime R2: one family, continuous severity."""

    def __init__(self, acfg):
        self.p = acfg["prob"]
        self.q = acfg["jpeg_quality"]
        self.s = acfg["resize_scale"]
        self.b = acfg["blur_sigma"]
        self.k = acfg["reencode_cycles"]

    def __call__(self, img, rng):
        if rng.random() >= self.p:
            return img
        family = rng.integers(4)
        if family == 0:
            return jpeg(img, rng.integers(self.q[0], self.q[1] + 1))
        if family == 1:
            return resize(img, rng.uniform(*self.s))
        if family == 2:
            return gaussian_blur(img, rng.uniform(*self.b))
        return reencode(img, rng.integers(self.k[0], self.k[1] + 1), rng)
