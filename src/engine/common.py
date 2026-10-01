"""Shared training plumbing: data loading (real or synthetic smoke data), AMP, checkpoint/resume, image grids."""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils import ROOT, load_json

MANIFEST_DIR = ROOT / "data" / "manifests"


# ------------------------------------------------------------------ smoke data (no dataset needed)
def synthetic_images(n: int, size: int = 128, seed: int = 0) -> np.ndarray:
    """Smooth random colour fields + shapes. Lets every script run end-to-end on CPU without the dataset."""
    import cv2
    rng = np.random.default_rng(seed)
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        low = rng.random((6, 6, 3)).astype(np.float32)
        img = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
        img = np.clip(img, 0, 1)
        cx, cy, r = rng.integers(20, size - 20, 2).tolist() + [int(rng.integers(8, 30))]
        cv2.circle(img, (cx, cy), r, rng.random(3).tolist(), -1)
        out[i] = (img * 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ Pets arrays + manifests
def pets_data(cfg: dict) -> dict:
    """Returns {'train','val','test'} uint8 arrays and {'val_items','test_items'} manifests."""
    if cfg.get("smoke"):
        from src.data.manifests import build_test_manifest, build_val_manifest
        tr, va, te = synthetic_images(64, seed=1), synthetic_images(32, seed=2), synthetic_images(4, seed=3)
        return {"train": tr, "val": va, "test": te,
                "val_items": build_val_manifest([f"v{i}" for i in range(len(va))])["items"],
                "test_items": build_test_manifest([f"t{i}" for i in range(len(te))])["items"]}
    from src.data.pets import load_arrays
    arrays = load_arrays()
    arrays["val_items"] = load_json(MANIFEST_DIR / "val.json")["items"]
    arrays["test_items"] = load_json(MANIFEST_DIR / "test.json")["items"]
    return arrays


def subset_array(arr: np.ndarray, frac: float, seed: int = 42) -> np.ndarray:
    """Deterministic fraction of the training images (used to make Optuna trials cheaper)."""
    if frac >= 1.0:
        return arr
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(arr), int(math.ceil(frac * len(arr))), replace=False))
    return arr[idx]


def subset_items(items: list, frac: float, seed: int = 42) -> list:
    if frac >= 1.0:
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), int(math.ceil(frac * len(items))), replace=False))
    return [items[i] for i in idx]


def make_loader(ds, batch_size=None, batch_sampler=None, shuffle=False, num_workers=2, device="cpu",
                drop_last=False):
    kw = dict(num_workers=num_workers, pin_memory=(str(device).startswith("cuda")),
              persistent_workers=num_workers > 0)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kw)


# ------------------------------------------------------------------ AMP
class Amp:
    """bf16 autocast on GPUs that support it (RTX 40xx do; no GradScaler needed), else fp16 + GradScaler, CPU off."""

    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = enabled and device.type == "cuda"
        self.dtype = torch.bfloat16 if (self.enabled and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and self.dtype == torch.float16)
        self.device = device

    def autocast(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, clip: float | None = None,
             params=None) -> None:
        self.scaler.scale(loss).backward()
        if clip is not None and params is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(optimizer)
        self.scaler.update()


# ------------------------------------------------------------------ misc
def cosine_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_steps: int = 0):
    total = max(1, epochs * steps_per_epoch)

    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def image_row_grid(rows: list[list[torch.Tensor]]) -> np.ndarray:
    """rows of CHW [0,1] tensors -> one uint8 HWC grid (rows x cols)."""
    r = [torch.cat([t.detach().float().cpu().clamp(0, 1) for t in row], dim=2) for row in rows]
    g = torch.cat(r, dim=1)
    return (g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def out_dir(cfg: dict) -> Path:
    p = ROOT / cfg.get("out_dir", "checkpoints/tmp")
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_limit(loader, max_batches: int | None):
    """Yield at most max_batches batches (smoke tests)."""
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield b


def subset_dataset(ds, n: int | None):
    return ds if n is None or n >= len(ds) else Subset(ds, list(range(n)))
