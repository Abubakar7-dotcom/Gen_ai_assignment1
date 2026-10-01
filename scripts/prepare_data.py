"""One-shot data preparation. Run from repo root:

    python -m scripts.prepare_data --pets          # download (if needed) + cache + split + manifests
    python -m scripts.prepare_data --fs2k          # cache + pairing check + stratified split (FS2K downloaded manually)
    python -m scripts.prepare_data --sanity        # save a visual grid of every corruption/severity

Small outputs (split JSONs, manifests, pairing report, sanity grid) are committed to git.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data import fs2k, pets
from src.data.manifests import build_test_manifest, build_val_manifest
from src.utils import ROOT, load_json, save_json

MANIFEST_DIR = ROOT / "data" / "manifests"


def prepare_pets(skip_download: bool = False) -> None:
    if not skip_download:
        pets.download()
    trainval_ids = pets.read_official_ids(pets.RAW_DIR, "trainval")
    test_ids = pets.read_official_ids(pets.RAW_DIR, "test")
    split = pets.make_split(trainval_ids)
    save_json(split, pets.SPLIT_PATH)
    print(f"split: train={len(split['train_idx'])} val={len(split['val_idx'])} test={len(test_ids)}")
    save_json(build_val_manifest(split["val_ids"]), MANIFEST_DIR / "val.json")
    save_json(build_test_manifest(test_ids), MANIFEST_DIR / "test.json")
    print("manifests written")
    pets.build_cache()


def prepare_fs2k() -> None:
    report = fs2k.build_cache_and_split()
    save_json(report, ROOT / "results" / "data" / "fs2k_pairing_report.json")
    print({k: (v if k == "split" else {"pairs": v["pairs"], "n_problems": v["n_problems"]})
           for k, v in report.items()})


def sanity_grid() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.corruptions import apply_corruption

    arrays = pets.load_arrays()
    items = load_json(MANIFEST_DIR / "test.json")["items"][:10]          # first test image, all 10 versions
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.6))
    for ax, it in zip(axes, items):
        ax.imshow(apply_corruption(arrays["test"][it["index"]], it["condition"], it["params"]))
        extra = f"\n{it['params'].get('area', 0):.0%}" if it["condition"] == "occlusion" else ""
        ax.set_title(f"{it['condition']}\n{it['severity']}{extra}", fontsize=8)
        ax.axis("off")
    out = ROOT / "results" / "data" / "corruption_sanity_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pets", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--fs2k", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args()
    if a.pets:
        prepare_pets(a.skip_download)
    if a.fs2k:
        prepare_fs2k()
    if a.sanity:
        sanity_grid()
