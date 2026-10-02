"""Optuna hyperparameter search for every task.

    python -m optuna_studies.tune --task t1 --trials 15
    python -m optuna_studies.tune --task t2_spec --trials 12      # shared search, run on the blur specialist
    python -m optuna_studies.tune --task t2_cls --trials 15
    python -m optuna_studies.tune --task t3 --trials 10
    python -m optuna_studies.tune --task t4 --trials 10

Design (documented in the report):
  * TPE sampler (seed 42) + MedianPruner (prunes a trial whose intermediate score is below the median of earlier
    trials at the same epoch). Short trials on a 50% training subset, then the best config is retrained in full.
  * Studies persist to SQLite (optuna_studies/<task>.db) -> a killed search resumes where it stopped.
  * CUDA out-of-memory inside a trial -> pruned (batch-size choices are capped for a 6 GB GPU).
Outputs: configs/<task>_best.yaml (used by train.train automatically), results/<task>/optuna/{best_params.json,
trials.csv, history.png, importance.png}.
"""
from __future__ import annotations

import argparse
import copy

import optuna
import torch
import yaml

from src.utils import ROOT, load_config, save_json

CHANNEL_SETS = {"small": [16, 32, 64, 128], "medium": [32, 64, 128, 256], "wide": [48, 96, 192, 384]}


def space_ae(t: optuna.Trial) -> dict:
    return {"lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": t.suggest_categorical("batch_size", [32, 64, 128]),
            "latent_ch": t.suggest_categorical("latent_ch", [8, 16, 32, 64]),
            "base": t.suggest_categorical("base", [24, 32, 48]),
            "dropout": t.suggest_float("dropout", 0.0, 0.3),
            "alpha": t.suggest_float("alpha", 0.5, 0.95)}


def space_cls(t: optuna.Trial) -> dict:
    return {"lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": t.suggest_categorical("batch_size", [32, 64, 128]),
            "channels": CHANNEL_SETS[t.suggest_categorical("channels", list(CHANNEL_SETS))],
            "dropout": t.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}


def space_moe(t: optuna.Trial) -> dict:
    l1 = t.suggest_float("l1_w", 0.5, 0.95)
    return {"joint_lr": t.suggest_float("joint_lr", 1e-5, 3e-4, log=True),
            "tau": t.suggest_float("tau", 0.5, 2.0),
            "ce_w": t.suggest_float("ce_w", 0.01, 0.5, log=True),
            "bal_w": t.suggest_float("bal_w", 1e-3, 0.1, log=True),
            "l1_w": l1, "ssim_w": 1.0 - l1}


def space_gan(t: optuna.Trial) -> dict:
    return {"g_lr": t.suggest_float("g_lr", 5e-5, 5e-4, log=True),
            "d_lr": t.suggest_float("d_lr", 5e-5, 5e-4, log=True),
            "batch_size": t.suggest_categorical("batch_size", [8, 16, 32]),
            "g_base": t.suggest_categorical("g_base", [32, 48, 64]),
            "dropout": t.suggest_categorical("dropout", [0.0, 0.25, 0.5]),
            "style_dim": t.suggest_categorical("style_dim", [8, 16, 32]),
            "lambda_l1": t.suggest_float("lambda_l1", 10.0, 200.0, log=True)}


# task -> (engine, search space, short-trial overrides, default #trials)
TASKS = {
    "t1": ("ae", space_ae, {"epochs": 4, "train_subset": 0.5, "val_subset": 0.5}, 15),
    "t2_spec": ("ae", space_ae, {"epochs": 4, "train_subset": 0.5, "conditions": ["blur"]}, 12),
    "t2_cls": ("cls", space_cls, {"epochs": 4, "train_subset": 0.5}, 15),
    "t3": ("moe", space_moe, {"warmup_epochs": 1, "joint_epochs": 3, "train_subset": 0.5, "val_subset": 0.5}, 10),
    "t4": ("gan", space_gan, {"epochs": 10}, 10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--trials", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    engine_name, space, short, n_default = TASKS[a.task]
    engine = __import__(f"src.engine.{engine_name}", fromlist=["run"])
    base = load_config(ROOT / "configs" / f"{a.task}.yaml")
    base.update(device=a.device, smoke=a.smoke, wandb=False)

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base)
        cfg.update(short)
        cfg.update(space(trial))
        if a.smoke:
            cfg.update({k: 1 for k in ("epochs", "warmup_epochs", "joint_epochs") if k in cfg})
        cfg["out_dir"] = f"checkpoints/optuna_{a.task}"
        try:
            return engine.run(cfg, trial=trial)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")

    storage = None if a.smoke else f"sqlite:///{(ROOT / 'optuna_studies' / f'{a.task}.db').as_posix()}"
    study = optuna.create_study(study_name=a.task, storage=storage, load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    done = len([t for t in study.trials if t.state.is_finished()])
    n_trials = 2 if a.smoke else max(0, (a.trials or n_default) - done)
    # hand cached GPU memory back after every trial: the 6 GB card is shared with other processes
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   callbacks=[lambda study, trial: torch.cuda.empty_cache()])

    out = ROOT / "results" / a.task / "optuna"
    out.mkdir(parents=True, exist_ok=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = {"task": a.task, "n_trials": len(study.trials), "n_complete": len(complete),
               "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
               "trial_overrides": short}
    if complete:
        summary.update(best_value=study.best_value, best_trial=study.best_trial.number, best_params=study.best_params)
        best_cfg = load_config(ROOT / "configs" / f"{a.task}.yaml")
        # apply the same transformation the objective used (e.g. channel-set names, ssim_w = 1 - l1_w)
        best_cfg.update(space(optuna.trial.FixedTrial(study.best_params)))
        if not a.smoke:
            with open(ROOT / "configs" / f"{a.task}_best.yaml", "w") as f:
                f.write(f"# generated by optuna_studies/tune.py from trial {study.best_trial.number} "
                        f"(objective={study.best_value:.4f})\n")
                yaml.safe_dump(best_cfg, f, sort_keys=False)
    save_json(summary, out / ("best_params_smoke.json" if a.smoke else "best_params.json"))
    if not a.smoke:
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)
        _plots(study, out)
    print(summary)


def _plots(study, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm
    for name, fn in (("history", ovm.plot_optimization_history), ("importance", ovm.plot_param_importances),
                     ("intermediate", ovm.plot_intermediate_values)):
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            fig.set_size_inches(7, 4.5); fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=140)
            plt.close("all")
        except Exception as e:                  # importance needs >= 2 complete trials
            print(f"plot {name} skipped: {e}")


if __name__ == "__main__":
    main()
