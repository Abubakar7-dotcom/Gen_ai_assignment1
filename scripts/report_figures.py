"""Collect the figures used by the LaTeX report into report/figures/ (so report/ compiles on its own, e.g. Overleaf).

    python -m scripts.report_figures

Tall example grids from eval/evaluate.py (one example per row) are re-tiled into two columns so they fit an
IEEE page. Rows are found from the white gaps between them; nothing is redrawn or edited. Photo grids are stored
as JPEG (quality 90) to keep the PDF small; plots and UI screenshots stay PNG. Google Stitch screenshots saved in
docs/stitch/ (01_universal.png ... 04_sketch.png) are copied too, and the report shows them automatically.
"""
import shutil

import numpy as np
from PIL import Image

from src.utils import ROOT

RES, OUT = ROOT / "results", ROOT / "report" / "figures"
COPY = {  # report name <- results path
    "sanity_grid.png": "data/corruption_sanity_grid.png",
    "t1_failures.jpg": "t1/failures.png",
    "t2_confusion.png": "t2/confusion_matrix.png",
    "t2_misrouting.jpg": "t2/misrouting_failures.png",
    "t3_heatmap.png": "t3/routing_heatmap.png",
    "t4_failures.jpg": "t4/failures.png",
    "t4_progress.jpg": "t4/progress.png",
    "ablation_skip.jpg": "ablations/t1_skip_examples.png",
    **{f"optuna_{t}_importance.png": f"{t}/optuna/importance.png" for t in ["t1", "t2_cls", "t2_spec", "t3", "t4"]},
}
RETILE = {"t1_examples.jpg": "t1/examples.png",
          "t4_examples.jpg": "t4/examples.png", "t3_dominant.jpg": "t3/dominant_vs_distributed.png"}
STITCH = ["01_universal", "02_hard", "03_moe", "04_sketch"]


def save(img: Image.Image, dst):
    img.convert("RGB").save(dst, quality=90, optimize=True) if dst.suffix == ".jpg" else img.save(dst)


def row_bands(img: np.ndarray) -> list[tuple[int, int]]:
    """(top, bottom) of each example row: an image band (mostly non-white) plus the caption text above it."""
    ink = (img < 245).any(axis=2).mean(axis=1)            # fraction of non-white pixels per pixel row
    tiles, start = [], None
    for y, v in enumerate(ink > 0.3):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start > 40:                            # an image band, not a line of text
                tiles.append((start, y))
            start = None
    gap = tiles[1][0] - tiles[0][1] if len(tiles) > 1 else 60     # caption height between two image bands
    return [(max(0, t - gap + 4), b + 4) for t, b in tiles]


def retile(src, dst, cols=2, keep=None):
    img = np.asarray(Image.open(src).convert("RGB"))
    rows = [img[t:b] for t, b in row_bands(img)]
    if keep is not None:                                  # subset of rows, e.g. two examples per style
        rows = [rows[i] for i in keep]
    h = max(r.shape[0] for r in rows)
    rows = [np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0)), constant_values=255) for r in rows]
    per_col = -(-len(rows) // cols)
    rows += [np.full_like(rows[0], 255)] * (per_col * cols - len(rows))
    columns = [np.vstack(rows[c * per_col:(c + 1) * per_col]) for c in range(cols)]
    spacer = np.full((columns[0].shape[0], 30, 3), 255, np.uint8)
    save(Image.fromarray(np.hstack([columns[0], spacer, *columns[1:]])), dst)
    print(f"{dst.name}: {len(row_bands(img))} rows -> {cols} columns")


def t4_progress(epochs=(5, 20, 45, 149), photos=(0, 3, 5), tile=128):
    """results/t4/progress.png from the fixed-validation sample grids saved during T4 training
    (checkpoints/t4/samples/epoch_XXX.png: rows photo | real | generated | as style 1 | 2 | 3, one column per photo).
    Only on the training machine (checkpoints are not in git); the PNG itself is committed."""
    src = ROOT / "checkpoints" / "t4" / "samples"
    if not src.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grids = {e: np.asarray(Image.open(src / f"epoch_{e:03d}.png").convert("RGB")) for e in epochs}
    crop = lambda g, r, c: g[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
    cols = ["photo", "real sketch"] + [f"epoch {e}" for e in epochs]
    fig, axes = plt.subplots(len(photos), len(cols), figsize=(1.15 * len(cols), 1.2 * len(photos)))
    for i, c in enumerate(photos):
        g0 = grids[epochs[0]]
        panels = [crop(g0, 0, c), crop(g0, 1, c)] + [crop(grids[e], 2, c) for e in epochs]
        for j, img in enumerate(panels):
            axes[i, j].imshow(img); axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(RES / "t4" / "progress.png", dpi=150)
    plt.close(fig)
    print("results/t4/progress.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t4_progress()
    for name, rel in COPY.items():
        save(Image.open(RES / rel), OUT / name)
    for s in STITCH:
        src = ROOT / "docs" / "stitch" / f"{s}.png"
        if src.exists():
            shutil.copyfile(src, OUT / f"stitch_{s}.png")
    for name, rel in RETILE.items():
        # T4 grid rows are 4 examples per style; the report shows the first two of each style
        retile(RES / rel, OUT / name, keep=[0, 4, 8, 1, 5, 9] if name.startswith("t4_") else None)


if __name__ == "__main__":
    main()
