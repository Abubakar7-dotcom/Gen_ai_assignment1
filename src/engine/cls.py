"""Corruption-classifier training (Task 2). Objective: validation macro-F1."""
from __future__ import annotations

import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F

from src.data.corruptions import CONDITIONS
from src.data.pets import ManifestDataset, RuntimeCorruptionDataset
from src.data.sampler import BalancedConditionBatchSampler
from src.engine.common import Amp, cosine_scheduler, make_loader, maybe_limit, out_dir, pets_data, subset_array
from src.metrics import classification_report
from src.models.classifier import CorruptionClassifier
from src.utils import Tracker, get_device, load_checkpoint, save_checkpoint, save_json, seed_everything


def build_model(cfg: dict) -> CorruptionClassifier:
    return CorruptionClassifier(channels=tuple(cfg["channels"]), dropout=cfg["dropout"])


@torch.no_grad()
def predict(model, loader, device, amp: Amp, max_batches=None):
    model.eval()
    ys, ps, idx = [], [], []
    for x_t, _, c, i in maybe_limit(loader, max_batches):
        with amp.autocast():
            logits = model(x_t.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), 1).cpu()); ys.append(c); idx.append(i)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(idx).numpy()


def run(cfg: dict, trial: optuna.Trial | None = None) -> float:
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    smoke = cfg.get("smoke", False)
    max_b = 2 if smoke else None
    data = pets_data(cfg)
    train_ds = RuntimeCorruptionDataset(subset_array(data["train"], cfg.get("train_subset", 1.0)))
    val_ds = ManifestDataset(data["val"], data["val_items"])
    nw = 0 if smoke else cfg.get("num_workers", 2)
    train_dl = make_loader(train_ds, batch_sampler=BalancedConditionBatchSampler(
        len(train_ds), cfg["batch_size"], seed=cfg.get("seed", 42)), num_workers=nw, device=device)
    val_dl = make_loader(val_ds, batch_size=128, num_workers=nw, device=device)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = cosine_scheduler(opt, cfg["epochs"], max(1, len(train_dl)), warmup_steps=cfg.get("warmup_steps", 100))
    amp = Amp(device, cfg.get("amp", True))
    odir = out_dir(cfg)
    start, best, run_id = 0, -1.0, None
    if trial is None and cfg.get("resume", True):
        ck = load_checkpoint(odir / "last.pt")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start, best, run_id = ck["epoch"] + 1, ck["best"], ck.get("run_id")
    tracker = Tracker(cfg.get("project", "genai-a1"), cfg.get("run_name", "t2-classifier"), cfg,
                      enabled=cfg.get("wandb", True) and trial is None and not smoke, resume_id=run_id)

    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x_t, _, c in maybe_limit(train_dl, max_b):
            x_t, c = x_t.to(device, non_blocking=True), c.to(device, non_blocking=True)
            with amp.autocast():
                logits = model(x_t)
            loss = F.cross_entropy(logits.float(), c, label_smoothing=cfg.get("label_smoothing", 0.0))
            opt.zero_grad(set_to_none=True)
            amp.step(loss, opt)
            sched.step()
            tot += loss.item() * len(c); correct += (logits.argmax(1) == c).sum().item(); n += len(c)
        y, p, _ = predict(model, val_dl, device, amp, max_b)
        rep = classification_report(y, p.argmax(1), names=CONDITIONS)
        log = {"epoch": epoch, "train/loss": tot / max(n, 1), "train/acc": correct / max(n, 1),
               "val/acc": rep["accuracy"], "val/macro_f1": rep["macro_f1"], "time/epoch_s": time.time() - t0,
               **{f"val/recall_{k}": v["recall"] for k, v in rep["per_class"].items()}}
        tracker.log(log, step=epoch)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in log.items()})
        improved = rep["macro_f1"] > best
        best = max(best, rep["macro_f1"])
        if trial is not None:
            trial.report(rep["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            continue
        if improved:
            save_checkpoint(odir / "best.pt", model=model.state_dict(), model_cfg=model.cfg, epoch=epoch, val=rep)
        save_checkpoint(odir / "last.pt", model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best=best, run_id=tracker.id, model_cfg=model.cfg)
    if trial is None:
        save_json({"best_val_macro_f1": best, "config": cfg}, odir / "summary.json")
        tracker.summary({"best_val_macro_f1": best})
        tracker.checkpoint(odir / "best.pt", cfg.get("run_name", "t2-classifier"), {"best_val_macro_f1": best})
    tracker.finish()
    return best
