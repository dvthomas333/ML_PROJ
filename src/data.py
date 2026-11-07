"""CIFAKE loading, integrity audit, de-duplication and stratified splitting.

    python -m src.data --config configs/experiment.yaml

Produces one .npz of uint8 arrays (x_train, y_train, x_val, y_val, x_test, y_test) and an audit .json.
Images are stored decoded so every later step (perturbation, transform) starts from identical pixels.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .utils import load_config, save_json

CLASSES = {"REAL": 0, "FAKE": 1}  # FAKE (Stable Diffusion) is the positive class
IMG_SIZE = 32
EXTS = {".jpg", ".jpeg", ".png"}


def scan_split(split_dir: Path):
    paths, labels = [], []
    for name, label in CLASSES.items():
        files = sorted(p for p in (split_dir / name).iterdir() if p.suffix.lower() in EXTS)
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels, dtype=np.int64)


def _qtable_id(im):
    q = getattr(im, "quantization", None)
    if not q:
        return None
    blob = json.dumps({str(k): list(v) for k, v in sorted(q.items())}).encode()
    return hashlib.md5(blob).hexdigest()[:8]


def decode_and_audit(paths, labels):
    """Decode every file to 32x32x3 uint8 and record storage metadata.

    Files that fail to decode or are not 32x32 are the dataset's "missing values": they are dropped and listed.
    """
    images, keep, meta, dropped = [], [], [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                info = {"format": im.format, "mode": im.mode, "bytes": p.stat().st_size, "qtable": _qtable_id(im)}
                arr = np.asarray(im.convert("RGB"))
        except (OSError, ValueError) as e:
            dropped.append({"path": str(p), "reason": repr(e)})
            continue
        if arr.shape != (IMG_SIZE, IMG_SIZE, 3):
            dropped.append({"path": str(p), "reason": f"shape {arr.shape}"})
            continue
        images.append(arr)
        keep.append(i)
        meta.append(info)
    return np.stack(images), labels[keep], meta, dropped


def summarise_storage(meta, labels):
    """Per-class storage statistics plus a file-size-only 'shortcut' AUC (Grommelt et al., 2025).

    An AUC far from 0.5 means file size alone separates the classes, i.e. a compression bias exists.
    """
    out = {}
    for name, c in CLASSES.items():
        m = [mm for mm, y in zip(meta, labels) if y == c]
        sizes = np.array([mm["bytes"] for mm in m])
        q = Counter(mm["qtable"] for mm in m)
        out[name] = {
            "n": len(m),
            "formats": dict(Counter(mm["format"] for mm in m)),
            "modes": dict(Counter(mm["mode"] for mm in m)),
            "bytes_mean": float(sizes.mean()),
            "bytes_std": float(sizes.std()),
            "distinct_qtables": len(q),
            "top_qtables": q.most_common(3),
        }
    sizes = np.array([mm["bytes"] for mm in meta])
    out["filesize_shortcut_auc"] = float(roc_auc_score(labels, sizes))
    return out


def pixel_hash(a):
    return hashlib.sha1(a.tobytes()).hexdigest()


def dhash(a):
    """64-bit difference hash; used only to *report* near-duplicates."""
    g = np.asarray(Image.fromarray(a).convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes().hex()


def dedupe(x, y):
    """Keep the first copy of each exact pixel duplicate; drop hashes that carry both labels."""
    hashes = [pixel_hash(a) for a in x]
    labels_per_hash = {}
    for h, lab in zip(hashes, y):
        labels_per_hash.setdefault(h, set()).add(int(lab))
    seen, keep = set(), []
    for i, h in enumerate(hashes):
        if h in seen or len(labels_per_hash[h]) > 1:
            continue
        seen.add(h)
        keep.append(i)
    n_conflict = sum(1 for s in labels_per_hash.values() if len(s) > 1)
    return np.asarray(keep), {"removed": len(x) - len(keep), "label_conflicts": n_conflict}


def main(cfg):
    root = Path(cfg["data"]["root"])
    report = {}
    arrays = {}
    for split in ("train", "test"):
        paths, labels = scan_split(root / split)
        x, y, meta, dropped = decode_and_audit(paths, labels)
        keep, dup = dedupe(x, y)
        report[split] = {
            "files": len(paths),
            "undecodable_or_wrong_size": dropped,
            "storage": summarise_storage(meta, y),
            "exact_duplicates": dup,
        }
        arrays[split] = (x[keep], y[keep])

    # Train/test leakage: exact copies are removed from test; dHash matches are only reported.
    x_tr, y_tr = arrays["train"]
    x_te, y_te = arrays["test"]
    train_px = {pixel_hash(a) for a in x_tr}
    train_dh = {dhash(a) for a in x_tr}
    leak = np.array([pixel_hash(a) in train_px for a in x_te])
    near = int(sum(dhash(a) in train_dh for a in x_te[~leak]))
    x_te, y_te = x_te[~leak], y_te[~leak]
    report["leakage"] = {"exact_train_test_overlap_removed": int(leak.sum()), "dhash_near_duplicates_reported": near}

    idx = np.arange(len(y_tr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=cfg["data"]["val_fraction"], stratify=y_tr, random_state=cfg["data"]["split_seed"]
    )
    out = {
        "x_train": x_tr[tr_idx], "y_train": y_tr[tr_idx],
        "x_val": x_tr[va_idx], "y_val": y_tr[va_idx],
        "x_test": x_te, "y_test": y_te,
    }
    report["final_counts"] = {
        k: {"n": int(len(out[f"y_{k}"])), "fake": int(out[f"y_{k}"].sum())} for k in ("train", "val", "test")
    }
    Path(cfg["data"]["processed"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg["data"]["processed"], **out)
    save_json(report, cfg["data"]["audit"])
    print(json.dumps(report["final_counts"], indent=2))
    print("file-size shortcut AUC (train):", report["train"]["storage"]["filesize_shortcut_auc"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    main(load_config(ap.parse_args().config))
