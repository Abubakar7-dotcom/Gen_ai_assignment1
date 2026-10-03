"""Shared helpers: seeding, device selection, config loading, checkpointing, experiment tracking."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, **state) -> None:
    """Atomic save: write to tmp then rename, so a killed process never leaves a corrupt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------- tracking
class Tracker:
    """Thin wrapper around Weights & Biases. Falls back to no-op when disabled/offline (e.g. CPU smoke tests)."""

    def __init__(self, project: str, name: str, config: dict, enabled: bool = True, group: str | None = None,
                 resume_id: str | None = None):
        self.run = None
        if enabled and os.environ.get("WANDB_MODE") != "disabled":
            import wandb
            self.run = wandb.init(project=project, name=name, config=config, group=group,
                                  id=resume_id, resume="allow" if resume_id else None, reinit=True)

    @property
    def id(self) -> str | None:
        return self.run.id if self.run else None

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run:
            self.run.log(data, step=step)

    def image(self, key: str, img, step: int | None = None, caption: str | None = None) -> None:
        if self.run:
            import wandb
            self.run.log({key: wandb.Image(img, caption=caption)}, step=step)

    def summary(self, data: dict) -> None:
        if self.run:
            self.run.summary.update(data)

    def checkpoint(self, path: str | Path, name: str, metadata: dict | None = None) -> None:
        """Log a checkpoint file as a W&B model artifact (alias 'best'), so checkpoints are tracked with the run."""
        if self.run and Path(path).exists():
            import wandb
            art = wandb.Artifact(name, type="model", metadata=metadata or {})
            art.add_file(str(path))
            self.run.log_artifact(art, aliases=["best"])

    def finish(self) -> None:
        if self.run:
            self.run.finish()
