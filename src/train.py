"""Train one model.

    python -m src.train --model dct --regime clean --seed 0
    python -m src.train --model spatial --regime augmented --seed 3 --params configs/tuned/spatial.json
    python -m src.train --model dct --regime clean --seed 0 --mask masks/mi_top256.npy   # feature-selection ablation
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .models import MODEL_NAMES, build_model, count_macs, count_params
from .perturbations import RandomPerturbation
from .utils import env_info, get_device, load_config, load_processed, predict, save_json, set_seed


class ArrayDataset(Dataset):
    def __init__(self, x, y, hflip=False, augment=None):
        self.x, self.y, self.hflip, self.augment = x, y, hflip, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.hflip and torch.rand(1).item() < 0.5:
            img = img[:, ::-1]
        if self.augment is not None:
            # numpy RNG derived from torch RNG, which DataLoader seeds per worker from the run seed
            img = self.augment(np.ascontiguousarray(img), np.random.default_rng(int(torch.randint(2**31 - 1, (1,)))))
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        return t, torch.tensor(float(self.y[i]))


def make_optimizer(model, lr, wd):
    decay = [p for p in model.parameters() if p.ndim > 1]  # conv / linear weights
    no_decay = [p for p in model.parameters() if p.ndim <= 1]  # BN affine, biases
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_one(cfg, model_name, regime, seed, out_dir, hp=None, mask=None, max_epochs=None, report=None, save=True):
    """Returns a summary dict. `report(epoch, val_auc)` is an optional hook used by the tuner for pruning."""
    hp = {**cfg["train"], **(hp or {})}
    epochs = max_epochs or hp["epochs"]
    set_seed(seed)
    device = get_device()
    data = load_processed(cfg)

    mask_t = None if mask is None else torch.as_tensor(np.load(mask) if isinstance(mask, (str, Path)) else mask)
    model = build_model(model_name, cfg, mask=mask_t, dropout=hp.get("dropout")).to(device)

    stats_loader = DataLoader(ArrayDataset(data["x_train"], data["y_train"]), batch_size=1024, shuffle=False)
    for rep in model.representations():
        rep.fit(stats_loader, device)

    augment = RandomPerturbation(cfg["augment"]) if regime == "augmented" else None
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ArrayDataset(data["x_train"], data["y_train"], hflip=hp["hflip"], augment=augment),
        batch_size=int(hp["batch_size"]), shuffle=True, generator=g, drop_last=True,
        num_workers=hp["num_workers"], pin_memory=device.type == "cuda", persistent_workers=hp["num_workers"] > 0,
    )
    opt = make_optimizer(model, hp["lr"], hp["weight_decay"])
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, hp["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch)
    )
    use_amp = hp["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # pos_weight = N_real / N_fake: exactly 1 for balanced CIFAKE, corrects any imbalance left by de-duplication
    n_fake = float(data["y_train"].sum())
    pos_weight = torch.tensor((len(data["y_train"]) - n_fake) / n_fake, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = Path(out_dir)
    best = {"val_loss": float("inf"), "epoch": -1, "val_auc": None}
    history, bad = [], 0
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb).float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)

        p_val = predict(model, data["x_val"], device)
        y_val = data["y_val"]
        p_clip = np.clip(p_val, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(y_val * np.log(p_clip) + (1 - y_val) * np.log(1 - p_clip)))
        val_auc = float(roc_auc_score(y_val, p_val))
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_auc": val_auc,
                        "lr": sched.get_last_lr()[0]})
        print(f"[{model_name}/{regime}/s{seed}] ep {epoch:02d} train {tot / n:.4f} val {val_loss:.4f} auc {val_auc:.4f}")
        if report is not None:
            report(epoch, val_auc)

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "val_auc": val_auc}
            bad = 0
            if save:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "model_name": model_name, "regime": regime,
                            "seed": seed, "cfg": cfg, "hp": hp, "epoch": epoch,
                            "mask": None if mask_t is None else mask_t.cpu().numpy()}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= hp["patience"]:
                break

    summary = {"model": model_name, "regime": regime, "seed": seed, "best": best, "hp": hp,
               "params": count_params(model), "macs": count_macs(model), "env": env_info()}
    if save:
        save_json(history, out_dir / "history.json")
        save_json(summary, out_dir / "summary.json")
    return summary


def run_name(model, regime, seed, mask=None):
    tag = f"_{Path(mask).stem}" if mask else ""
    return f"{model}_{regime}{tag}_s{seed}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--regime", choices=("clean", "augmented"), default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", help="JSON of tuned hyper-parameters (from src.tune)")
    ap.add_argument("--mask", help="npy coefficient mask (1x3x32x32) for the feature-selection ablation")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    cfg = load_config(a.config)
    hp = json.load(open(a.params)) if a.params else None
    out = Path(a.runs) / run_name(a.model, a.regime, a.seed, a.mask)
    train_one(cfg, a.model, a.regime, a.seed, out, hp=hp, mask=a.mask)
