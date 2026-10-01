"""Training CLI for all tasks (full runs: checkpoints every epoch, resumable, W&B logging).

    python -m train.train --task t1
    python -m train.train --task t2_cls
    python -m train.train --task t2_spec --specialist blur        # salt_pepper | blur | occlusion
    python -m train.train --task t3
    python -m train.train --task t4
Options: --config (default: configs/<task>_best.yaml if it exists, else configs/<task>.yaml), --smoke (CPU, 2 batches,
synthetic data), --set key=value (override), --no-wandb, --fresh (ignore last.pt).
"""
from __future__ import annotations

import argparse

import yaml

from src.utils import ROOT, load_config

ENGINES = {"t1": "ae", "t2_spec": "ae", "t2_cls": "cls", "t3": "moe", "t4": "gan"}


def default_config(task: str):
    best = ROOT / "configs" / f"{task}_best.yaml"
    return best if best.exists() else ROOT / "configs" / f"{task}.yaml"


def parse_set(pairs: list[str]) -> dict:
    return {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(ENGINES))
    ap.add_argument("--config")
    ap.add_argument("--specialist", choices=["salt_pepper", "blur", "occlusion"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load_config(a.config or default_config(a.task))
    cfg.update(parse_set(a.set))
    cfg.update(device=a.device, smoke=a.smoke, wandb=not a.no_wandb, resume=not a.fresh, task=a.task)
    if a.task == "t2_spec":
        if not a.specialist:
            ap.error("--specialist is required for t2_spec")
        cfg["conditions"] = [a.specialist]
        cfg["out_dir"] = f"checkpoints/t2_spec_{a.specialist}"
        cfg["run_name"] = f"t2-specialist-{a.specialist}"
    if a.smoke:
        cfg["out_dir"] = f"checkpoints/smoke_{a.task}"
        for k in ("epochs", "joint_epochs"):
            if k in cfg:
                cfg[k] = 1
        cfg["warmup_epochs"] = 1 if "warmup_epochs" in cfg else None
        cfg = {k: v for k, v in cfg.items() if v is not None}

    engine = __import__(f"src.engine.{ENGINES[a.task]}", fromlist=["run"])
    score = engine.run(cfg)
    print(f"done: best validation objective = {score:.4f}")


if __name__ == "__main__":
    main()
