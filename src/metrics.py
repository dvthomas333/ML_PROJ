"""Classification and robustness metrics. FAKE = positive class."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),  # true-positive rate on generated images
        "specificity": tn / max(tn + fp, 1),  # true-negative rate on real images
        "f1": f1_score(y, yhat, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "mcc": matthews_corrcoef(y, yhat),
        "auc": roc_auc_score(y, p),
        "pred_fake_rate": float(yhat.mean()),  # exposes threshold drift under perturbation
    }


def fast_auc(y, s):
    """Mann-Whitney AUC (ties averaged); used inside the bootstrap where sklearn is too slow."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def robustness_summary(df):
    """Per (run, family): mean AUC over severity levels (mAUC), drop and relative robustness vs clean.

    df columns: run, model, regime, seed, family, level, auc, ...
    """
    clean = df[df.family == "clean"].set_index("run")["auc"]
    fam = df[df.family != "clean"].groupby(["run", "model", "regime", "seed", "family"], as_index=False)["auc"].mean()
    fam = fam.rename(columns={"auc": "mAUC"})
    fam["clean_auc"] = fam["run"].map(clean)
    fam["auc_drop"] = fam["clean_auc"] - fam["mAUC"]
    fam["relative_robustness"] = (fam["mAUC"] - 0.5) / (fam["clean_auc"] - 0.5)  # chance-corrected
    return fam
