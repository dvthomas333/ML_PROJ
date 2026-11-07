"""Paired, seed-averaged bootstrap test of Delta-AUC between two models on every test condition.

    python -m src.stats --a dct --b spatial --regime clean

For each bootstrap replicate one resample of test-image indices is shared by all seeds and both models,
so the comparison is paired at the image level. Holm-Bonferroni controls the family-wise error over conditions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import fast_auc
from .perturbations import condition_grid
from .utils import load_config


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load_preds(runs_dir, model, regime, seeds, cond):
    return np.stack([np.load(Path(runs_dir) / f"{model}_{regime}_s{s}" / "preds" / f"{cond}.npy") for s in seeds])


def compare(cfg, a, b, regime, runs_dir, results_dir):
    y = np.load(Path(results_dir) / "y_test.npy")
    seeds = cfg["evaluation"]["seeds"]
    B = cfg["evaluation"]["bootstrap"]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]

    rows = []
    for c in condition_grid():
        pa, pb = load_preds(runs_dir, a, regime, seeds, c.name), load_preds(runs_dir, b, regime, seeds, c.name)
        observed = np.mean([fast_auc(y, pa[s]) - fast_auc(y, pb[s]) for s in range(len(seeds))])
        boot = np.array([np.mean([fast_auc(y[i], pa[s][i]) - fast_auc(y[i], pb[s][i]) for s in range(len(seeds))])
                         for i in idx])
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"condition": c.name, "family": c.family, "level": c.level, "delta_auc": observed,
                     "ci_low": lo, "ci_high": hi, "p_boot": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_boot"])
    df["significant"] = df["p_holm"] < cfg["evaluation"]["alpha"]
    out = Path(results_dir) / f"paired_{a}_vs_{b}_{regime}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--a", default="dct")
    ap.add_argument("--b", default="spatial")
    ap.add_argument("--regime", default="clean")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    x = ap.parse_args()
    compare(load_config(x.config), x.a, x.b, x.regime, x.runs, x.results)
