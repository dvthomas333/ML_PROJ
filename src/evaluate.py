"""Robustness evaluation of every trained run on the 27-condition test grid, plus efficiency profiling.

    python -m src.evaluate                 # metrics for all runs/*/best.pt
    python -m src.evaluate --efficiency    # parameters, MACs, CPU latency per architecture
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import binary_metrics, robustness_summary
from .models import MODEL_NAMES, build_model, count_macs, count_params, cpu_latency_ms
from .perturbations import apply_condition, condition_grid
from .utils import env_info, get_device, load_checkpoint, load_config, load_processed, predict, save_json


def perturbed_test_set(cfg, x_test, cond):
    """Perturbed copies of the test set are generated once and cached, so every model sees identical pixels."""
    path = Path(cfg["evaluation"]["cache_dir"]) / f"test_{cond.name}.npy"
    if path.exists():
        return np.load(path)
    x = apply_condition(x_test, cond, cfg["evaluation"]["perturb_seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def evaluate_runs(cfg, runs_dir, results_dir):
    device = get_device()
    data = load_processed(cfg)
    y = data["y_test"]
    grid = condition_grid()
    test_sets = {c.name: perturbed_test_set(cfg, data["x_test"], c) for c in grid}

    rows = []
    for ckpt_path in sorted(Path(runs_dir).glob("*/best.pt")):
        run_dir = ckpt_path.parent
        model, ckpt = load_checkpoint(ckpt_path, device)
        (run_dir / "preds").mkdir(exist_ok=True)
        for c in grid:
            p = predict(model, test_sets[c.name], device)
            np.save(run_dir / "preds" / f"{c.name}.npy", p.astype(np.float32))
            rows.append({"run": run_dir.name, "model": ckpt["model_name"], "regime": ckpt["regime"],
                         "masked": ckpt.get("mask") is not None, "seed": ckpt["seed"],
                         "condition": c.name, "family": c.family, "level": c.level, **binary_metrics(y, p)})
        print("evaluated", run_dir.name)

    df = pd.DataFrame(rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "y_test.npy", y)
    df.to_csv(results_dir / "metrics_per_run.csv", index=False)

    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "auc"]
    agg = df.groupby(["model", "regime", "masked", "condition", "family", "level"])[metric_cols].agg(["mean", "std"])
    agg.to_csv(results_dir / "metrics_mean_std.csv")
    robustness_summary(df).to_csv(results_dir / "robustness_per_run.csv", index=False)
    save_json(env_info(), results_dir / "environment.json")


def efficiency(cfg, results_dir):
    rows = []
    for name in MODEL_NAMES:
        m = build_model(name, cfg)
        med, p95 = cpu_latency_ms(m, threads=1)
        med_all, _ = cpu_latency_ms(m, threads=torch.get_num_threads())
        rows.append({"model": name, "params": count_params(m), "macs": count_macs(m),
                     "size_mb_fp32": count_params(m) * 4 / 2**20,
                     "cpu_ms_1thread_median": med, "cpu_ms_1thread_p95": p95, "cpu_ms_allthreads_median": med_all})
    df = pd.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(results_dir) / "efficiency.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--efficiency", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.efficiency:
        efficiency(cfg, a.results)
    else:
        evaluate_runs(cfg, a.runs, a.results)
