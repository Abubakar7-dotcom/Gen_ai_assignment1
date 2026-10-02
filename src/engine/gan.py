"""Style-conditioned pix2pix training (Task 4). Objective for Optuna: validation score from L1 + SSIM."""
from __future__ import annotations

import time

import numpy as np
import optuna
import torch

from src.data.fs2k import PairedSketchDataset, load_fs2k
from src.engine.common import Amp, image_row_grid, make_loader, maybe_limit, out_dir, synthetic_images
from src.losses import d_loss, g_loss
from src.metrics import ssim_per_image
from src.models.pix2pix import StylePatchDiscriminator, StyleUNetGenerator, init_weights
from src.utils import ROOT, Tracker, get_device, load_checkpoint, save_checkpoint, save_json, seed_everything


def fs2k_data(cfg: dict):
    if cfg.get("smoke"):
        p, s = synthetic_images(24, seed=5), synthetic_images(24, seed=6)
        st = np.arange(24) % 3
        return (p[:16], s[:16], st[:16]), (p[16:], s[16:], st[16:])
    return load_fs2k("train"), load_fs2k("val")


@torch.no_grad()
def evaluate(G, loader, device, max_batches=None) -> dict:
    G.eval()
    l1s, sss, sts = [], [], []
    for x, y, s in maybe_limit(loader, max_batches):
        x, y, s = x.to(device), y.to(device), s.to(device)
        fake = G(x, s).float()
        f01, y01 = (fake + 1) / 2, (y + 1) / 2
        l1s.append((f01 - y01).abs().flatten(1).mean(1).cpu()); sss.append(ssim_per_image(f01, y01).cpu())
        sts.append(s.cpu())
    l1s, sss, sts = (torch.cat(t).numpy() for t in (l1s, sss, sts))
    out = {"l1": float(l1s.mean()), "ssim": float(sss.mean())}
    out["score"] = 0.5 * (1 - min(out["l1"], 1.0)) + 0.5 * out["ssim"]       # higher is better
    for k in range(3):
        if (sts == k).any():
            out[f"ssim_style{k + 1}"] = float(sss[sts == k].mean())
    return out


def sample_grid(G, fixed, device) -> np.ndarray:
    """Rows: photo | real sketch | generated (true style) | generated as style 1, 2, 3 (fixed val photos)."""
    G.eval()
    x, y, s = (t.to(device) for t in fixed)
    with torch.no_grad():
        rows = [list((x + 1) / 2), list((y + 1) / 2), list((G(x, s).float() + 1) / 2)]
        for k in range(3):
            rows.append(list((G(x, torch.full_like(s, k)).float() + 1) / 2))
    return image_row_grid(rows)


def run(cfg: dict, trial: optuna.Trial | None = None) -> float:
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    smoke = cfg.get("smoke", False)
    max_b = 2 if smoke else None
    (tp, ts, tst), (vp, vs, vst) = fs2k_data(cfg)
    nw = 0 if smoke else cfg.get("num_workers", 2)
    train_dl = make_loader(PairedSketchDataset(tp, ts, tst, augment=True), batch_size=cfg["batch_size"],
                           shuffle=True, drop_last=True, num_workers=nw, device=device)
    val_dl = make_loader(PairedSketchDataset(vp, vs, vst), batch_size=32, num_workers=nw, device=device)
    fixed_idx = list(range(min(6, len(vp))))
    fixed = next(iter(make_loader(torch.utils.data.Subset(PairedSketchDataset(vp, vs, vst), fixed_idx),
                                  batch_size=len(fixed_idx), num_workers=0)))

    G = StyleUNetGenerator(base=cfg["g_base"], style_dim=cfg["style_dim"], dropout=cfg["dropout"]).to(device)
    D = StylePatchDiscriminator(base=cfg.get("d_base", 64), style_dim=cfg["style_dim"]).to(device)
    G.apply(init_weights); D.apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), lr=cfg["g_lr"], betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=cfg["d_lr"], betas=(0.5, 0.999))
    epochs = cfg["epochs"]
    decay_start = cfg.get("decay_start", epochs // 2)                 # pix2pix: constant lr, then linear decay
    lam = lambda e: 1.0 if e < decay_start else max(0.0, 1 - (e - decay_start) / max(1, epochs - decay_start))
    schG, schD = (torch.optim.lr_scheduler.LambdaLR(o, lam) for o in (optG, optD))
    ampG, ampD = Amp(device, cfg.get("amp", True)), Amp(device, cfg.get("amp", True))

    odir = out_dir(cfg)
    start, best, run_id = 0, -1.0, None
    if trial is None and cfg.get("resume", True):
        ck = load_checkpoint(odir / "last.pt")
        if ck:
            G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); optG.load_state_dict(ck["optG"])
            optD.load_state_dict(ck["optD"]); schG.load_state_dict(ck["schG"]); schD.load_state_dict(ck["schD"])
            start, best, run_id = ck["epoch"] + 1, ck["best"], ck.get("run_id")
            print(f"resumed from epoch {start}")
    tracker = Tracker(cfg.get("project", "genai-a1"), cfg.get("run_name", "t4-cgan"), cfg,
                      enabled=cfg.get("wandb", True) and trial is None and not smoke, resume_id=run_id)

    for epoch in range(start, epochs):
        G.train(); D.train()
        t0, acc, n = time.time(), {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}, 0
        for x, y, s in maybe_limit(train_dl, max_b):
            x, y, s = x.to(device, non_blocking=True), y.to(device, non_blocking=True), s.to(device)
            # ---- D step: real pairs -> 1, generated pairs -> 0
            with ampD.autocast():
                fake = G(x, s)
                ld, pd = d_loss(D(x, y, s).float(), D(x, fake.detach(), s).float())
            optD.zero_grad(set_to_none=True)
            ampD.step(ld, optD)
            # ---- G step: fool D + stay close to the paired sketch
            with ampG.autocast():
                fake_logits = D(x, fake, s)
            lg, pg = g_loss(fake_logits.float(), fake.float(), y, cfg["lambda_l1"])
            optG.zero_grad(set_to_none=True)
            ampG.step(lg, optG)
            for k, v in {**pd, **pg}.items():
                acc[k] += v
            n += 1
        schG.step(); schD.step()
        val = evaluate(G, val_dl, device, max_b)
        log = {"epoch": epoch, "time/epoch_s": time.time() - t0, **{f"train/{k}": v / max(n, 1) for k, v in acc.items()},
               **{f"val/{k}": v for k, v in val.items()}}
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
            save_checkpoint(odir / "best_G.pt", G=G.state_dict(), g_cfg=G.cfg, epoch=epoch, val=val)
        save_checkpoint(odir / "last.pt", G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(),
                        optD=optD.state_dict(), schG=schG.state_dict(), schD=schD.state_dict(), epoch=epoch,
                        best=best, run_id=tracker.id, g_cfg=G.cfg)
        if epoch % cfg.get("sample_every", 5) == 0 or epoch == epochs - 1:
            grid = sample_grid(G, fixed, device)
            tracker.image("samples/grid", grid, step=epoch,
                          caption="rows: photo | real | generated | as style 1 | as style 2 | as style 3")
            if not smoke:
                import cv2
                (odir / "samples").mkdir(exist_ok=True)
                cv2.imwrite(str(odir / "samples" / f"epoch_{epoch:03d}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if trial is None:
        save_json({"best_val_score": best, "config": cfg}, odir / "summary.json")
    tracker.finish()
    return best
