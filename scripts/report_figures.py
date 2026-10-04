"""Collect the figures used by the LaTeX report into report/figures/ (so report/ compiles on its own, e.g. Overleaf).

    python -m scripts.report_figures

Tall example grids from eval/evaluate.py (one example per row) are re-tiled into two columns so they fit an
IEEE page. Rows are found from the white gaps between them; nothing is redrawn or edited.
"""
import shutil

import numpy as np
from PIL import Image

from src.utils import ROOT

RES, OUT = ROOT / "results", ROOT / "report" / "figures"
COPY = {  # report name <- results path
    "sanity_grid.png": "data/corruption_sanity_grid.png",
    "t1_failures.png": "t1/failures.png",
    "t2_confusion.png": "t2/confusion_matrix.png",
    "t2_misrouting.png": "t2/misrouting_failures.png",
    "t3_heatmap.png": "t3/routing_heatmap.png",
    "t3_failures.png": "t3/failures.png",
    "t4_failures.png": "t4/failures.png",
    "ablation_skip.png": "ablations/t1_skip_examples.png",
    **{f"optuna_{t}_{k}.png": f"{t}/optuna/{k}.png" for t in ["t1", "t2_cls", "t2_spec", "t3", "t4"]
       for k in ["history", "importance"]},
}
RETILE = {"t1_examples.png": "t1/examples.png", "t2_examples.png": "t2/examples.png",
          "t3_examples.png": "t3/examples.png", "t4_examples.png": "t4/examples.png",
          "t3_dominant.png": "t3/dominant_vs_distributed.png"}


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


def retile(src, dst, cols=2):
    img = np.asarray(Image.open(src).convert("RGB"))
    rows = [img[t:b] for t, b in row_bands(img)]
    h = max(r.shape[0] for r in rows)
    rows = [np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0)), constant_values=255) for r in rows]
    per_col = -(-len(rows) // cols)
    rows += [np.full_like(rows[0], 255)] * (per_col * cols - len(rows))
    columns = [np.vstack(rows[c * per_col:(c + 1) * per_col]) for c in range(cols)]
    spacer = np.full((columns[0].shape[0], 30, 3), 255, np.uint8)
    Image.fromarray(np.hstack([columns[0], spacer, *columns[1:]])).save(dst)
    print(f"{dst.name}: {len(row_bands(img))} rows -> {cols} columns")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rel in COPY.items():
        shutil.copyfile(RES / rel, OUT / name)
    for name, rel in RETILE.items():
        retile(RES / rel, OUT / name)


if __name__ == "__main__":
    main()
