"""Training / validation curves for the report, from results/curves_history.json (scripts/export_wandb_history.py).

    python -m eval.plot_curves        # -> report/figures/curves_t{1,2,3,4}.pdf

One quantity per axis (no dual axes). Series keep a fixed colour per condition across all figures and also differ
in dash style, so the curves stay readable in greyscale print.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import ROOT  # noqa: E402

OUT = ROOT / "report" / "figures"
# fixed categorical order (validated for colour-vision deficiency); identity of a condition never changes colour
COLOR = {"clean": "#2a78d6", "salt_pepper": "#eb6834", "blur": "#1baf7a", "occlusion": "#eda100"}
DASH = {"clean": "-", "salt_pepper": "--", "blur": "-.", "occlusion": ":"}
NAME = {"clean": "clean", "salt_pepper": "salt & pepper", "blur": "blur", "occlusion": "occlusion"}
INK, GRID = "#333333", "#e6e6e6"

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.4,
                     "axes.edgecolor": "#888888", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"})


def series(rows, key):
    pts = [(r["epoch"], r[key]) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def line(ax, rows, key, label, color, dash="-"):
    x, y = series(rows, key)
    ax.plot(x, y, dash, color=color, label=label)


def legend_below(ax, ncol):
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.3), handlelength=2.4)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = json.load(open(ROOT / "results" / "curves_history.json"))

    # ---- Task 1: training loss | validation PSNR per condition
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), layout="constrained")
    line(a, h["t1"], "train/loss", "train loss", COLOR["clean"])
    a.set(title="(a) T1 training loss  $\\alpha L_1+(1-\\alpha)(1-\\mathrm{SSIM})$", xlabel="epoch", ylabel="loss")
    for c in COLOR:
        line(b, h["t1"], f"val/psnr_{c}", NAME[c], COLOR[c], DASH[c])
    b.set(title="(b) T1 validation PSNR by input condition", xlabel="epoch", ylabel="PSNR (dB)")
    legend_below(b, 4)
    save(fig, "curves_t1.pdf")

    # ---- Task 2: classifier accuracy | specialist validation PSNR
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), layout="constrained")
    line(a, h["t2_cls"], "train/acc", "train accuracy", COLOR["clean"])
    line(a, h["t2_cls"], "val/acc", "validation accuracy", COLOR["salt_pepper"], "--")
    a.set(title="(a) T2 corruption classifier", xlabel="epoch", ylabel="accuracy")
    a.legend(loc="lower right")
    for c in ["salt_pepper", "blur", "occlusion"]:
        line(b, h[f"t2_spec_{c}"], "val/psnr", f"{NAME[c]} specialist", COLOR[c], DASH[c])
    b.set(title="(b) T2 specialists, validation PSNR on own corruption", xlabel="epoch", ylabel="PSNR (dB)")
    b.legend(loc="lower right")
    save(fig, "curves_t2.pdf")

    # ---- Task 3: validation objective | mean routing weight per branch (warm-up shaded)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), layout="constrained")
    for ax in (a, b):
        ax.axvspan(-0.5, 2.5, color="#f1f1f1", zorder=0, lw=0)
        ax.text(1, 0.02, "warm-up\n(gate only)", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6, color="#666666")
    line(a, h["t3"], "val/score", "objective", COLOR["clean"])
    a.set(title="(a) T3 validation objective", xlabel="epoch", ylabel="objective")
    for c in COLOR:
        line(b, h["t3"], f"val/mean_w_{c}", "identity" if c == "clean" else f"{NAME[c]} expert", COLOR[c], DASH[c])
    b.axhline(0.25, color="#999999", lw=0.8)
    b.set(title="(b) T3 mean gate weight per branch (validation)", xlabel="epoch", ylabel="mean weight",
          ylim=(0, 0.45))
    legend_below(b, 4)
    save(fig, "curves_t3.pdf")

    # ---- Task 4: adversarial losses | L1 (train vs val) | validation SSIM per style
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.2), layout="constrained")
    line(a, h["t4"], "train/d_real", "D real", COLOR["clean"])
    line(a, h["t4"], "train/d_fake", "D fake", COLOR["salt_pepper"], "--")
    line(a, h["t4"], "train/g_adv", "G adversarial", COLOR["blur"], "-.")
    a.set(title="(a) T4 adversarial losses (BCE)", xlabel="epoch", ylabel="loss")
    a.legend(loc="center right")
    line(b, h["t4"], "train/g_l1", "train", COLOR["clean"])
    line(b, h["t4"], "val/l1", "validation", COLOR["salt_pepper"], "--")
    b.set(title="(b) T4 L1 reconstruction", xlabel="epoch", ylabel="L1")
    b.legend(loc="upper right")
    for s, col, dash in [(1, COLOR["clean"], "-"), (2, COLOR["salt_pepper"], "--"), (3, COLOR["blur"], "-.")]:
        line(c, h["t4"], f"val/ssim_style{s}", f"style {s}", col, dash)
    c.set(title="(c) T4 validation SSIM per style", xlabel="epoch", ylabel="SSIM")
    legend_below(c, 3)
    save(fig, "curves_t4.pdf")


if __name__ == "__main__":
    main()
