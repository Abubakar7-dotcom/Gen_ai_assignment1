"""Autoencoder training: Task 1 (universal, all 4 conditions) and Task 2 specialists (one corruption each).

Used by train/train_ae.py (full runs, checkpoints, W&B) and optuna_studies/tune_ae.py (trials, pruning).
"""
from __future__ import annotations

import time

import numpy as np
import optuna
import torch

from src.data.corruptions import CONDITIONS
from src.data.pets import ManifestDataset, RuntimeCorruptionDataset
from src.data.sampler import BalancedConditionBatchSampler
from src.engine.common import (Amp, cosine_scheduler, image_row_grid, make_loader, maybe_limit, out_dir, pets_data,
                               subset_array, subset_items)
from src.losses import recon_loss
from src.metrics import psnr_per_image, restoration_score, ssim_per_image
from src.models.autoencoder import DenoisingAE
from src.utils import Tracker, get_device, load_checkpoint, save_checkpoint, save_json, seed_everything


def build_model(cfg: dict) -> DenoisingAE:
    return DenoisingAE(base=cfg["base"], latent_ch=cfg["latent_ch"], dropout=cfg["dropout"],
                       skip=cfg.get("skip", False))


@torch.no_grad()
def evaluate(model, loader, device, amp: Amp, max_batches=None) -> dict:
    """Mean PSNR/SSIM overall and per condition on a manifest loader."""
    model.eval()
    ps, ss, cs = [], [], []
    for x_t, x, c, _ in maybe_limit(loader, max_batches):
        x_t, x = x_t.to(device, non_blocking=True), x.to(device, non_blocking=True)
        with amp.autocast():
            x_hat = model(x_t)
        ps.append(psnr_per_image(x_hat, x).cpu()); ss.append(ssim_per_image(x_hat, x).cpu()); cs.append(c)
    ps, ss, cs = torch.cat(ps).numpy(), torch.cat(ss).numpy(), torch.cat(cs).numpy()
    out = {"psnr": float(ps.mean()), "ssim": float(ss.mean())}
    out["score"] = restoration_score(out["psnr"], out["ssim"])
    for k, name in enumerate(CONDITIONS):
        m = cs == k
        if m.any():
            out[f"psnr_{name}"], out[f"ssim_{name}"] = float(ps[m].mean()), float(ss[m].mean())
    return out


def run(cfg: dict, trial: optuna.Trial | None = None) -> float:
    """Train one autoencoder. Returns best validation restoration score (higher is better)."""
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    conditions = cfg.get("conditions", CONDITIONS)
    smoke = cfg.get("smoke", False)
    max_b = 2 if smoke else None

    data = pets_data(cfg)
    train_imgs = subset_array(data["train"], cfg.get("train_subset", 1.0))
    val_items = subset_items(data["val_items"], cfg.get("val_subset", 1.0))
    train_ds = RuntimeCorruptionDataset(train_imgs, conditions=conditions)
    val_ds = ManifestDataset(data["val"], val_items, conditions=conditions)
    nw = 0 if smoke else cfg.get("num_workers", 2)
    if len(conditions) == len(CONDITIONS):           # universal: exactly balanced batches
        train_dl = make_loader(train_ds, batch_sampler=BalancedConditionBatchSampler(
            len(train_ds), cfg["batch_size"], seed=cfg.get("seed", 42)), num_workers=nw, device=device)
    else:                                            # specialist: single corruption type
        train_dl = make_loader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                               num_workers=nw, device=device)
    val_dl = make_loader(val_ds, batch_size=64, num_workers=nw, device=device)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg["epochs"]
    sched = cosine_scheduler(opt, epochs, max(1, len(train_dl)), warmup_steps=cfg.get("warmup_steps", 100))
    amp = Amp(device, cfg.get("amp", True))

    odir = out_dir(cfg)
    start_epoch, best, run_id = 0, -1.0, None
    if trial is None and cfg.get("resume", True):
        ck = load_checkpoint(odir / "last.pt")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_epoch, best, run_id = ck["epoch"] + 1, ck["best"], ck.get("run_id")
            print(f"resumed from epoch {start_epoch}")
    tracker = Tracker(cfg.get("project", "genai-a1"), cfg.get("run_name", "ae"), cfg,
                      enabled=cfg.get("wandb", True) and trial is None and not smoke, group=cfg.get("group"),
                      resume_id=run_id)

    fixed = next(iter(val_dl))                       # same val images every epoch for visual progress
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x_t, x, _ in maybe_limit(train_dl, max_b):
            x_t, x = x_t.to(device, non_blocking=True), x.to(device, non_blocking=True)
            with amp.autocast():
                x_hat = model(x_t)
            loss, parts = recon_loss(x_hat, x, cfg["alpha"])
            opt.zero_grad(set_to_none=True)
            amp.step(loss, opt)
            sched.step()
            tot += loss.item(); n += 1
        val = evaluate(model, val_dl, device, amp, max_b)
        log = {"epoch": epoch, "train/loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"],
               "time/epoch_s": time.time() - t0, **{f"val/{k}": v for k, v in val.items()}}
        tracker.log(log, step=epoch)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in log.items()})

        improved = val["score"] > best
        best = max(best, val["score"])
        if trial is not None:
            trial.report(val["score"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            continue
        if improved:
            save_checkpoint(odir / "best.pt", model=model.state_dict(), model_cfg=model.cfg, epoch=epoch,
                            val=val, conditions=conditions)
        save_checkpoint(odir / "last.pt", model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best=best, run_id=tracker.id, model_cfg=model.cfg)
        if epoch % cfg.get("sample_every", 5) == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                xt, xc = fixed[0][:6].to(device), fixed[1][:6].to(device)
                xh = model(xt).float()
            grid = image_row_grid([list(xc), list(xt), list(xh), list((xh - xc).abs() * 3)])
            tracker.image("samples/grid", grid, step=epoch, caption="rows: clean | input | output | abs error x3")

    if trial is None:
        save_json({"best_val_score": best, "config": cfg}, odir / "summary.json")
        tracker.summary({"best_val_score": best})
        tracker.checkpoint(odir / "best.pt", cfg.get("run_name", "ae"), {"best_val_score": best})
    tracker.finish()
    return best
