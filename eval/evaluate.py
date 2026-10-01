"""Final test-set evaluation + report figures for every task (uses the untouched official test set).

    python -m eval.evaluate --task t1     # universal DAE
    python -m eval.evaluate --task t2     # classifier report + oracle vs predicted routing
    python -m eval.evaluate --task t3     # soft MoE + routing analysis
    python -m eval.evaluate --task t4     # face-to-sketch
Outputs go to results/<task>/: per_item.csv, summary.csv/.md, *.png figures, metrics.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.corruptions import CONDITIONS
from src.data.pets import ManifestDataset
from src.engine.common import make_loader, pets_data
from src.metrics import classification_report, psnr_per_image, ssim_per_image
from src.utils import ROOT, get_device, save_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEV_ORDER = ["none", "low", "medium", "high"]


# ----------------------------------------------------------------- shared helpers
def run_restoration(fn, data, device, batch=64, extra_cols=None):
    """Apply fn(x_tilde, labels) -> (x_hat, extras dict of [B,...] tensors) over the whole test manifest."""
    ds = ManifestDataset(data["test"], data["test_items"])
    rows, outs = [], {}
    for x_t, x, c, idx in make_loader(ds, batch_size=batch, num_workers=0):
        x_t, x, c = x_t.to(device), x.to(device), c.to(device)
        with torch.no_grad():
            x_hat, extras = fn(x_t, c)
        p_in, s_in = psnr_per_image(x_t, x).cpu(), ssim_per_image(x_t, x).cpu()
        p_out, s_out = psnr_per_image(x_hat, x).cpu(), ssim_per_image(x_hat, x).cpu()
        for j, i in enumerate(idx.tolist()):
            it = ds.items[i]
            r = {"item": i, "id": it["id"], "condition": it["condition"], "severity": it["severity"],
                 "psnr_in": float(p_in[j]), "ssim_in": float(s_in[j]),
                 "psnr_out": float(p_out[j]), "ssim_out": float(s_out[j])}
            for k, v in extras.items():
                v = v[j].detach().cpu()
                if v.ndim == 0:
                    r[k] = v.item()
                else:
                    r.update({f"{k}_{n}": float(e) for n, e in zip(CONDITIONS, v.tolist())})
            rows.append(r)
    return pd.DataFrame(rows), ds


def summarize(df: pd.DataFrame, cols: list[str], out: Path, name="summary") -> pd.DataFrame:
    df = df.copy()
    df["severity"] = pd.Categorical(df["severity"], SEV_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], CONDITIONS, ordered=True)
    by_cs = df.groupby(["condition", "severity"], observed=True)[cols].mean().round(4)
    by_c = df.groupby("condition", observed=True)[cols].mean().round(4)
    overall = df[cols].mean().round(4).to_frame("overall").T
    by_cs.to_csv(out / f"{name}_by_condition_severity.csv"); by_c.to_csv(out / f"{name}_by_condition.csv")
    with open(out / f"{name}.md", "w") as f:
        f.write("## Overall\n\n" + overall.to_markdown() + "\n\n## By condition\n\n" + by_c.to_markdown() +
                "\n\n## By condition and severity\n\n" + by_cs.to_markdown() + "\n")
    print(by_cs)
    return by_cs


def _err_map(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return (a - b).abs().mean(0).cpu().numpy()


def example_grid(ds, df, items, restore_fn, device, path, title, annot=None):
    """Rows = examples; cols = clean | corrupted | output | |error|."""
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2.05 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(items):
        x_t, x, c, _ = ds[i]
        with torch.no_grad():
            x_hat, _ = restore_fn(x_t[None].to(device), torch.tensor([c], device=device))
        x_hat = x_hat[0].float().cpu()
        row = df[df["item"] == i].iloc[0]
        panels = [(x, "clean"), (x_t, f"{row['condition']} / {row['severity']}\nSSIM {row['ssim_in']:.3f}"),
                  (x_hat, f"output\nSSIM {row['ssim_out']:.3f}" + (f"\n{annot(row)}" if annot else "")),
                  (None, "|error|")]
        for k, (img, t) in enumerate(panels):
            ax = axes[r, k]
            if img is None:
                ax.imshow(_err_map(x_hat, x), cmap="inferno", vmin=0, vmax=0.5)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(t, fontsize=7); ax.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pick_examples(df: pd.DataFrame, n_per=1, seed=0) -> list[int]:
    """12 examples: one per (condition, severity) for the 3 corruptions (9) + 3 clean, distinct images."""
    rng = np.random.default_rng(seed)
    chosen, used = [], set()
    groups = [(c, s) for c in CONDITIONS[1:] for s in ["low", "medium", "high"]] + [("clean", "none")] * 3
    for c, s in groups:
        cand = df[(df.condition == c) & (df.severity == s) & (~df.id.isin(used))]
        if cand.empty:                                    # tiny smoke datasets: allow repeated images
            cand = df[(df.condition == c) & (df.severity == s)]
        r = cand.iloc[int(rng.integers(len(cand)))]
        chosen.append(int(r["item"])); used.add(r["id"])
    return chosen


def pick_failures(df: pd.DataFrame, n=4, key="ssim_out") -> list[int]:
    d = df[df.condition != "clean"].sort_values(key)
    out, used = [], set()
    for _, r in d.iterrows():
        if r["id"] not in used:
            out.append(int(r["item"])); used.add(r["id"])
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------- tasks
def eval_t1(data, device, out, smoke):
    from src.loaders import load_ae
    from src.models.autoencoder import DenoisingAE
    model = (DenoisingAE(base=8, latent_ch=8).eval() if smoke else load_ae("t1")).to(device)
    fn = lambda x, c: (model(x).float(), {})
    df, ds = run_restoration(fn, data, device)
    df.to_csv(out / "per_item.csv", index=False)
    summarize(df, ["psnr_in", "psnr_out", "ssim_in", "ssim_out"], out)
    example_grid(ds, df, pick_examples(df), fn, device, out / "examples.png", "Task 1: universal DAE (test set)")
    example_grid(ds, df, pick_failures(df), fn, device, out / "failures.png", "Task 1: lowest-SSIM failure cases")


def eval_t2(data, device, out, smoke):
    from src.loaders import HardRouter, load_classifier, load_specialists
    if smoke:
        from src.models.autoencoder import DenoisingAE
        from src.models.classifier import CorruptionClassifier
        router = HardRouter(CorruptionClassifier((8, 8, 8, 8)), {s: DenoisingAE(8, 8) for s in
                                                                  ["salt_pepper", "blur", "occlusion"]})
    else:
        router = HardRouter(load_classifier(), load_specialists())
    router = router.eval().to(device)
    pred_fn = lambda x, c: (lambda o: (o[0].float(), {"pred": o[2], "probs": o[1]}))(router(x))
    oracle_fn = lambda x, c: (router(x, route=c)[0].float(), {})
    df_p, ds = run_restoration(pred_fn, data, device)
    df_o, _ = run_restoration(oracle_fn, data, device)
    df = df_p.rename(columns={"psnr_out": "psnr_pred", "ssim_out": "ssim_pred"})
    df["psnr_oracle"], df["ssim_oracle"] = df_o["psnr_out"].values, df_o["ssim_out"].values
    df["true"] = df["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    df["misrouted"] = df["pred"] != df["true"]
    df.to_csv(out / "per_item.csv", index=False)

    rep = classification_report(df["true"], df["pred"], names=CONDITIONS)
    save_json(rep, out / "classifier_metrics.json")
    _confusion_plot(np.array(rep["confusion_matrix_normalized"]), out / "confusion_matrix.png")
    mis = df.groupby(["condition", "severity"])["misrouted"].mean().round(4)
    mis.to_csv(out / "misrouting_rate.csv")
    summarize(df, ["psnr_in", "psnr_oracle", "psnr_pred", "ssim_in", "ssim_oracle", "ssim_pred"], out)
    print("classifier acc", rep["accuracy"], "macro F1", rep["macro_f1"])

    df["ssim_out"] = df["ssim_pred"]
    annot = lambda r: f"pred={CONDITIONS[int(r['pred'])]}"
    example_grid(ds, df, pick_examples(df), pred_fn, device, out / "examples.png",
                 "Task 2: hard routing (predicted)", annot)
    bad = df[df.misrouted].assign(gap=lambda d: d.ssim_oracle - d.ssim_pred).sort_values("gap", ascending=False)
    items = bad.drop_duplicates("id")["item"].head(4).astype(int).tolist()
    if items:
        example_grid(ds, df, items, pred_fn, device, out / "misrouting_failures.png",
                     "Task 2: failures caused by classifier errors", annot)


def _confusion_plot(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CONDITIONS, rotation=30, ha="right"); ax.set_yticks(range(4), CONDITIONS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im, fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def eval_t3(data, device, out, smoke):
    from src.loaders import load_moe
    if smoke:
        from src.models.autoencoder import DenoisingAE
        from src.models.classifier import CorruptionClassifier
        from src.models.moe import SoftMoE
        moe = SoftMoE(CorruptionClassifier((8, 8, 8, 8)), [DenoisingAE(8, 8) for _ in range(3)])
    else:
        moe = load_moe()
    moe = moe.eval().to(device)
    fn = lambda x, c: (lambda o: (o[0].float(), {"w": o[1].float()}))(moe(x))
    df, ds = run_restoration(fn, data, device)
    wcols = [f"w_{c}" for c in CONDITIONS]
    W = df[wcols].values
    df["entropy"] = -(W * np.log(W + 1e-9)).sum(1)
    df["top_branch"] = W.argmax(1)
    df.to_csv(out / "per_item.csv", index=False)
    summarize(df, ["psnr_in", "psnr_out", "ssim_in", "ssim_out"], out)
    heat = summarize(df, wcols, out, name="routing_weights")
    _heatmap(heat, out / "routing_heatmap.png")
    usage = {"mean_weight": dict(zip(CONDITIONS, W.mean(0).round(4).tolist())),
             "argmax_share": dict(zip(CONDITIONS, (np.bincount(W.argmax(1), minlength=4) / len(W)).round(4).tolist())),
             "gate_accuracy_vs_true_label": float((df["top_branch"] == df["condition"].map(
                 {c: i for i, c in enumerate(CONDITIONS)})).mean())}
    usage["inactive_branches"] = [c for c, s in usage["argmax_share"].items() if s < 0.02]
    save_json(usage, out / "expert_usage.json"); print(usage)
    annot = lambda r: " ".join(f"{c[:4]}:{r[f'w_{c}']:.2f}" for c in CONDITIONS)
    dom = df[df.condition != "clean"].sort_values("entropy").drop_duplicates("id")["item"].head(3).astype(int).tolist()
    dist = df.sort_values("entropy", ascending=False).drop_duplicates("id")["item"].head(3).astype(int).tolist()
    example_grid(ds, df, dom + dist, fn, device, out / "dominant_vs_distributed.png",
                 "Task 3: dominant (top 3) vs distributed (bottom 3) routing", annot)
    example_grid(ds, df, pick_examples(df), fn, device, out / "examples.png", "Task 3: soft MoE (test set)", annot)
    example_grid(ds, df, pick_failures(df), fn, device, out / "failures.png", "Task 3: failure cases", annot)


def _heatmap(heat: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["identity", "A_salt", "A_blur", "A_occ"])
    ax.set_yticks(range(len(heat)), [f"{c} / {s}" for c, s in heat.index])
    for i in range(heat.shape[0]):
        for j in range(4):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black" if v > .5 else "white", fontsize=8)
    ax.set_xlabel("branch"); ax.set_ylabel("true corruption / severity"); ax.set_title("Mean routing weight")
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def eval_t4(device, out, smoke):
    from src.data.fs2k import PairedSketchDataset, load_fs2k
    from src.engine.common import synthetic_images
    from src.loaders import load_generator
    from src.models.pix2pix import StyleUNetGenerator
    if smoke:
        G = StyleUNetGenerator(base=8, style_dim=4)
        p, s, st = synthetic_images(12, seed=7), synthetic_images(12, seed=8), np.arange(12) % 3
    else:
        G = load_generator()
        p, s, st = load_fs2k("test")
    G = G.eval().to(device)
    ds = PairedSketchDataset(p, s, st)
    rows, fakes = [], []
    for x, y, sty in make_loader(ds, batch_size=32):
        with torch.no_grad():
            f = G(x.to(device), sty.to(device)).float().cpu()
        f01, y01 = (f + 1) / 2, (y + 1) / 2
        l1 = (f01 - y01).abs().flatten(1).mean(1); ss = ssim_per_image(f01, y01)
        for j in range(len(x)):
            rows.append({"item": len(rows), "style": int(sty[j]) + 1, "l1": float(l1[j]), "ssim": float(ss[j])})
        fakes.append(f01)
    df = pd.DataFrame(rows); df.to_csv(out / "per_item.csv", index=False)
    summ = pd.concat([df[["l1", "ssim"]].mean().to_frame("overall").T,
                      df.groupby("style")[["l1", "ssim"]].mean()]).round(4)
    summ.to_csv(out / "summary.csv"); open(out / "summary.md", "w").write(summ.to_markdown()); print(summ)
    fakes = torch.cat(fakes)
    rng = np.random.default_rng(0)
    for name, items in (("examples", [int(rng.choice(df[df["style"] == k]["item"])) for k in (1, 2, 3)
                                      for _ in range(4)]),
                        ("failures", df.sort_values("ssim")["item"].head(4).tolist())):
        fig, axes = plt.subplots(len(items), 4, figsize=(8, 2.05 * len(items)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(items):
            x, y, sty = ds[i]
            with torch.no_grad():
                alt = [G(x[None].to(device), torch.tensor([k], device=device)).float().cpu()[0] for k in range(3)]
            panels = [((x + 1) / 2, "photo"), ((y + 1) / 2, f"real (style {sty + 1})"),
                      (fakes[i], f"generated\nSSIM {df.loc[i, 'ssim']:.3f}"),
                      ((alt[(sty + 1) % 3] + 1) / 2, f"same photo as style {(sty + 1) % 3 + 1}")]
            for k, (img, t) in enumerate(panels):
                axes[r, k].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy()); axes[r, k].set_title(t, fontsize=7)
                axes[r, k].axis("off")
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = get_device(a.device)
    out = ROOT / "results" / (f"smoke_{a.task}" if a.smoke else a.task)
    out.mkdir(parents=True, exist_ok=True)
    if a.task == "t4":
        return eval_t4(device, out, a.smoke)
    data = pets_data({"smoke": a.smoke})
    {"t1": eval_t1, "t2": eval_t2, "t3": eval_t3}[a.task](data, device, out, a.smoke)


if __name__ == "__main__":
    main()
