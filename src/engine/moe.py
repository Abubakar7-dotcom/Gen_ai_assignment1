"""Soft mixture-of-experts training (Task 3).

Stage 1 (warm-up): gate initialised from the Task-2 classifier, experts initialised from the Task-2 specialists and
FROZEN; only the gate trains. Stage 2 (joint): everything unfrozen, smaller learning rate.
Routing collapse = one branch receives > collapse_thr of the mean weight over the whole validation set.
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
from src.losses import moe_loss
from src.metrics import psnr_per_image, restoration_score, ssim_per_image
from src.models.autoencoder import DenoisingAE
from src.models.classifier import CorruptionClassifier
from src.models.moe import SoftMoE
from src.utils import ROOT, Tracker, get_device, load_checkpoint, save_checkpoint, save_json, seed_everything

EXPERTS = ["salt_pepper", "blur", "occlusion"]


def build_moe(cfg: dict, load_pretrained: bool = True) -> SoftMoE:
    """Construct gate + experts from Task-2 checkpoints (or random weights in smoke mode)."""
    if load_pretrained and not cfg.get("smoke"):
        ck = load_checkpoint(ROOT / cfg["classifier_ckpt"])
        gate = CorruptionClassifier(**{k: (tuple(v) if k == "channels" else v)
                                       for k, v in ck["model_cfg"].items()})
        gate.load_state_dict(ck["model"])
        experts = []
        for name in EXPERTS:
            ek = load_checkpoint(ROOT / cfg["specialist_ckpts"][name])
            e = DenoisingAE(**ek["model_cfg"]); e.load_state_dict(ek["model"]); experts.append(e)
    else:
        gate = CorruptionClassifier(channels=(16, 32, 64, 64), dropout=0.1)
        experts = [DenoisingAE(base=8, latent_ch=8) for _ in EXPERTS]
    return SoftMoE(gate, experts, tau=cfg["tau"])


@torch.no_grad()
def evaluate(moe: SoftMoE, loader, device, amp: Amp, max_batches=None) -> dict:
    moe.eval()
    ps, ss, ws, cs = [], [], [], []
    for x_t, x, c, _ in maybe_limit(loader, max_batches):
        x_t, x = x_t.to(device), x.to(device)
        with amp.autocast():
            x_hat, w, _ = moe(x_t)
        ps.append(psnr_per_image(x_hat, x).cpu()); ss.append(ssim_per_image(x_hat, x).cpu())
        ws.append(w.float().cpu()); cs.append(c)
    ps, ss, ws, cs = (torch.cat(t).numpy() for t in (ps, ss, ws, cs))
    out = {"psnr": float(ps.mean()), "ssim": float(ss.mean()), "gate_acc": float((ws.argmax(1) == cs).mean())}
    out["score"] = restoration_score(out["psnr"], out["ssim"])
    mean_w = ws.mean(0)
    out["max_branch_share"] = float(mean_w.max())
    out["mean_w_by_class"] = {CONDITIONS[k]: ws[cs == k].mean(0).round(4).tolist() for k in range(4) if (cs == k).any()}
    for b in range(4):
        out[f"mean_w_{CONDITIONS[b]}"] = float(mean_w[b])
    return out


def run(cfg: dict, trial: optuna.Trial | None = None) -> float:
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    smoke = cfg.get("smoke", False)
    max_b = 2 if smoke else None
    data = pets_data(cfg)
    train_ds = RuntimeCorruptionDataset(subset_array(data["train"], cfg.get("train_subset", 1.0)))
    val_ds = ManifestDataset(data["val"], subset_items(data["val_items"], cfg.get("val_subset", 1.0)))
    nw = 0 if smoke else cfg.get("num_workers", 2)
    train_dl = make_loader(train_ds, batch_sampler=BalancedConditionBatchSampler(
        len(train_ds), cfg["batch_size"], seed=cfg.get("seed", 42)), num_workers=nw, device=device)
    val_dl = make_loader(val_ds, batch_size=32, num_workers=nw, device=device)

    moe = build_moe(cfg).to(device)
    amp = Amp(device, cfg.get("amp", True))
    odir = out_dir(cfg)
    tracker = Tracker(cfg.get("project", "genai-a1"), cfg.get("run_name", "t3-moe"), cfg,
                      enabled=cfg.get("wandb", True) and trial is None and not smoke)
    loss_w = dict(l1_w=cfg["l1_w"], ssim_w=cfg["ssim_w"], ce_w=cfg["ce_w"], bal_w=cfg["bal_w"])

    warm, joint = cfg["warmup_epochs"], cfg["joint_epochs"]
    best, step = -1.0, 0
    fixed = next(iter(val_dl))
    for stage, n_ep, lr in (("warmup", warm, cfg["gate_lr"]), ("joint", joint, cfg["joint_lr"])):
        moe.train()
        moe.set_experts_trainable(stage == "joint")
        params = [p for p in moe.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
        sched = cosine_scheduler(opt, n_ep, max(1, len(train_dl)))
        for ep in range(n_ep):
            moe.train(); moe.set_experts_trainable(stage == "joint")
            t0, tot, n = time.time(), {}, 0
            for x_t, x, c in maybe_limit(train_dl, max_b):
                x_t, x, c = x_t.to(device), x.to(device), c.to(device)
                with amp.autocast():
                    x_hat, w, logits = moe(x_t)
                loss, parts = moe_loss(x_hat, x, w, logits, c, **loss_w)
                opt.zero_grad(set_to_none=True)
                amp.step(loss, opt, clip=1.0, params=params)
                sched.step()
                for k, v in parts.items():
                    tot[k] = tot.get(k, 0.0) + v
                n += 1
            val = evaluate(moe, val_dl, device, amp, max_b)
            log = {"stage": stage, "epoch": step, "time/epoch_s": time.time() - t0,
                   **{f"train/{k}": v / max(n, 1) for k, v in tot.items()},
                   **{f"val/{k}": v for k, v in val.items() if not isinstance(v, dict)}}
            tracker.log(log, step=step)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in log.items()})
            collapsed = val["max_branch_share"] > cfg.get("collapse_thr", 0.9)
            if trial is not None:
                trial.set_user_attr("max_branch_share", val["max_branch_share"])
                trial.report(val["score"], step)
                if collapsed or trial.should_prune():
                    raise optuna.TrialPruned()
            if val["score"] > best and not collapsed:
                best = val["score"]
                if trial is None:
                    save_checkpoint(odir / "best.pt", moe=moe.state_dict(), tau=moe.tau, epoch=step, val=val,
                                    gate_cfg=moe.gate.cfg, expert_cfgs=[e.cfg for e in moe.experts])
            if trial is None:
                save_checkpoint(odir / "last.pt", moe=moe.state_dict(), tau=moe.tau, epoch=step, best=best,
                                gate_cfg=moe.gate.cfg, expert_cfgs=[e.cfg for e in moe.experts])
                moe.eval()
                with torch.no_grad():
                    xh, w, _ = moe(fixed[0][:6].to(device))
                tracker.image("samples/grid", image_row_grid(
                    [list(fixed[1][:6]), list(fixed[0][:6]), list(xh.float())]), step=step,
                    caption="rows: clean | input | output")
            step += 1
    if trial is None:
        save_json({"best_val_score": best, "config": cfg}, odir / "summary.json")
    tracker.finish()
    return best
