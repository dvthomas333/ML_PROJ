"""How post-processing changes what each model relies on.

    python -m src.interpret --spatial-run runs/spatial_clean_s0 --freq-run runs/dct_clean_s0

Spatial model : Grad-CAM (last conv block) on clean vs perturbed copies of the same images.
Frequency model: input-gradient saliency |d logit / d z| over the normalised spectral input z.
Stability = per-image Spearman correlation between the clean and perturbed attribution maps.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .evaluate import perturbed_test_set
from .perturbations import condition_grid
from .spectral_analysis import radial_bands, stratified_subsample
from .utils import get_device, load_checkpoint, load_config, load_processed, to_tensor


def gradcam(model, x):
    acts = {}
    layer = model.backbone.gradcam_layer
    h = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    model.zero_grad(set_to_none=True)
    logit = model(x)
    a = acts["a"]
    (g,) = torch.autograd.grad(logit.sum(), a)
    h.remove()
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    return (cam / (cam.flatten(1).max(1).values.view(-1, 1, 1) + 1e-12)).detach().cpu().numpy()


def spectral_saliency(model, x):
    with torch.no_grad():
        z = model.rep(x)
    z.requires_grad_(True)
    logit = model.forward_from_rep(z)
    (g,) = torch.autograd.grad(logit.sum(), z)
    return g.abs().mean(1).cpu().numpy()  # N x 32 x 32 (mean over colour channels)


def batched(fn, model, x_uint8, device, bs=256):
    return np.concatenate([fn(model, to_tensor(x_uint8[i : i + bs]).to(device)) for i in range(0, len(x_uint8), bs)])


def stability(a, b):
    # nanmean: an all-zero Grad-CAM map has undefined rank correlation
    return float(np.nanmean([spearmanr(p.ravel(), q.ravel())[0] for p, q in zip(a, b)]))


def main(cfg, spatial_run, freq_run, results_dir):
    device = get_device()
    data = load_processed(cfg)
    idx = stratified_subsample(data["y_test"], cfg["analysis"]["interpret_images"], seed=2)
    ms, _ = load_checkpoint(Path(spatial_run) / "best.pt", device)
    mf, _ = load_checkpoint(Path(freq_run) / "best.pt", device)
    bands = radial_bands()

    clean = data["x_test"][idx]
    cam_clean = batched(gradcam, ms, clean, device)
    sal_clean = batched(spectral_saliency, mf, clean, device)
    rows, mean_sal = [], {}
    for c in condition_grid():
        x = perturbed_test_set(cfg, data["x_test"], c)[idx]
        cam = batched(gradcam, ms, x, device)
        sal = batched(spectral_saliency, mf, x, device)
        m = sal.mean(0)
        mean_sal[c.name] = m
        rows.append({
            "condition": c.name, "family": c.family, "level": c.level,
            "gradcam_stability": stability(cam_clean, cam),
            "saliency_stability": stability(sal_clean, sal),
            **{f"saliency_share_{k}": float(m[b].sum() / m.sum()) for k, b in bands.items()},
        })
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "attribution_stability.csv", index=False)
    np.savez(out / "mean_spectral_saliency.npz", **mean_sal)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--spatial-run", default="runs/spatial_clean_s0")
    ap.add_argument("--freq-run", default="runs/dct_clean_s0")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    main(load_config(a.config), a.spatial_run, a.freq_run, a.results)
