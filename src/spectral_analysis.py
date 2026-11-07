"""Model-agnostic spectral analysis and the mutual-information feature ranking.

    python -m src.spectral_analysis

1. Per-coefficient Cohen's d between REAL and FAKE log|DCT| on the clean training split.
2. Spectral Discriminability Index (SDI = mean |d|) on the test split under every perturbation, overall and
   per radial band, and its retention ratio SDI_c / SDI_clean.
3. Mutual information I(coefficient; label) on the training split -> ranking -> top-k masks for the
   feature-selection ablation (masks/mi_top{k}.npy).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif

from .evaluate import perturbed_test_set
from .perturbations import condition_grid
from .representations import Representation
from .utils import load_config, load_processed, to_tensor

SIZE = 32


def stratified_subsample(y, n, seed=0):
    rng = np.random.default_rng(seed)
    per = n // 2
    return np.concatenate([rng.choice(np.flatnonzero(y == c), min(per, (y == c).sum()), replace=False) for c in (0, 1)])


def log_dct(x_uint8, eps, batch=2048):
    rep = Representation("dct", eps=eps)
    with torch.no_grad():
        return np.concatenate([rep.transform(to_tensor(x_uint8[i : i + batch])).numpy()
                               for i in range(0, len(x_uint8), batch)])  # N x 3 x 32 x 32


def cohens_d(f, y):
    a, b = f[y == 1], f[y == 0]
    pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-12
    return (a.mean(0) - b.mean(0)) / pooled


def radial_bands():
    """Normalised DCT index radius rho = sqrt(u^2 + v^2) / (31 sqrt 2): low < 1/3 <= mid < 2/3 <= high."""
    u, v = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    rho = np.sqrt(u**2 + v**2) / ((SIZE - 1) * np.sqrt(2))
    return {"low": rho < 1 / 3, "mid": (rho >= 1 / 3) & (rho < 2 / 3), "high": rho >= 2 / 3}


def sdi(d, bands):
    a = np.abs(d)  # 3 x 32 x 32
    return {"all": float(a.mean()), **{k: float(a[:, m].mean()) for k, m in bands.items()}}


def main(cfg, results_dir, mask_dir):
    an, eps = cfg["analysis"], cfg["representation"]["log_eps"]
    data = load_processed(cfg)
    results_dir, mask_dir = Path(results_dir), Path(mask_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    bands = radial_bands()

    tr = stratified_subsample(data["y_train"], an["spectral_subsample"])
    f_tr, y_tr = log_dct(data["x_train"][tr], eps), data["y_train"][tr]
    d_train = cohens_d(f_tr, y_tr)
    np.save(results_dir / "cohens_d_train.npy", d_train)

    # Feature ranking by mutual information (training split only -> no test leakage).
    mi_idx = tr[stratified_subsample(y_tr, an["mi_subsample"], seed=1)] if len(tr) > an["mi_subsample"] else tr
    f_mi = log_dct(data["x_train"][mi_idx], eps).reshape(len(mi_idx), -1)
    mi = mutual_info_classif(f_mi, data["y_train"][mi_idx], discrete_features=False, random_state=0)
    np.save(results_dir / "mutual_information.npy", mi.reshape(3, SIZE, SIZE))
    order = np.argsort(-mi)
    for k in an["mi_topk"]:
        m = np.zeros(mi.size, dtype=np.float32)
        m[order[:k]] = 1.0
        np.save(mask_dir / f"mi_top{k}.npy", m.reshape(1, 3, SIZE, SIZE))

    # SDI under every test condition.
    te = stratified_subsample(data["y_test"], an["spectral_subsample"])
    rows, d_maps = [], {}
    for c in condition_grid():
        x = perturbed_test_set(cfg, data["x_test"], c)[te]
        d = cohens_d(log_dct(x, eps), data["y_test"][te])
        d_maps[c.name] = d
        rows.append({"condition": c.name, "family": c.family, "level": c.level, **sdi(d, bands)})
    df = pd.DataFrame(rows)
    clean = df[df.condition == "clean"].iloc[0]
    for k in ("all", "low", "mid", "high"):
        df[f"retention_{k}"] = df[k] / clean[k]
    df.to_csv(results_dir / "spectral_sdi.csv", index=False)

    show = ["clean", "jpeg_50", "resize_0.5", "blur_1", "reencode_10"]
    vmax = np.abs(d_maps["clean"]).mean(0).max()
    fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3))
    for ax, name in zip(axes, show):
        im = ax.imshow(np.abs(d_maps[name]).mean(0), cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("v")
        ax.set_ylabel("u")
    fig.colorbar(im, ax=axes, shrink=0.8, label="|Cohen's d| (mean over RGB)")
    fig.savefig(results_dir / "figures" / "spectral_discriminability.pdf", bbox_inches="tight")
    print(df[["condition", "all", "low", "mid", "high", "retention_all"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--masks", default="masks")
    a = ap.parse_args()
    main(load_config(a.config), a.results, a.masks)
