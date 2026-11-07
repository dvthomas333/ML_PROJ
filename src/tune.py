"""Equal-budget hyper-parameter search (Optuna TPE + median pruning) on the CLEAN validation split.

    python -m src.tune --model dct

Every model gets the same number of trials and epochs, and the objective never sees perturbed data,
so robustness is not tuned for and no model is favoured by a larger search.
"""
import argparse
import tempfile
from pathlib import Path

import optuna

from .models import MODEL_NAMES
from .train import train_one
from .utils import load_config, save_json


def main(cfg, model_name, out):
    t, space = cfg["tuning"], cfg["tuning"]["search_space"]

    def objective(trial):
        hp = {
            "lr": trial.suggest_float("lr", *space["lr"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
            "dropout": trial.suggest_categorical("dropout", space["dropout"]),
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        }

        def report(epoch, auc):
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with tempfile.TemporaryDirectory() as tmp:
            s = train_one(cfg, model_name, "clean", seed=0, out_dir=tmp, hp=hp,
                          max_epochs=t["epochs"], report=report, save=False)
        return s["best"]["val_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=t["trials"])
    save_json(study.best_params, Path(out) / f"{model_name}.json")
    study.trials_dataframe().to_csv(Path(out) / f"{model_name}_trials.csv", index=False)
    print(model_name, study.best_value, study.best_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--out", default="configs/tuned")
    a = ap.parse_args()
    main(load_config(a.config), a.model, a.out)
