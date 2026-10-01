"""FS2K paired photo -> sketch dataset (Task 4).

Expected layout under ``data/raw/FS2K`` (official release, https://github.com/DengPingFan/FS2K):
    photo/photo{1,2,3}/image####.<ext>
    sketch/sketch{1,2,3}/sketch####.<ext>
    anno_train.json, anno_test.json      # list of dicts with "image_name" (e.g. "photo1/image0110") and "style"
Extensions differ between sub-folders, so files are resolved by stem, never by assuming ".jpg".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.pets import load_rgb_128, to_tensor
from src.utils import ROOT, load_json, save_json

FS2K_DIR = ROOT / "data" / "raw" / "FS2K"
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_PATH = ROOT / "data" / "splits" / "fs2k_split.json"
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find(stem_path: Path) -> Path | None:
    for ext in IMG_EXTS:
        for cand in (stem_path.with_suffix(ext), stem_path.with_suffix(ext.upper())):
            if cand.exists():
                return cand
    return None


def resolve_pairs(root: Path, split: str) -> tuple[list[dict], list[str]]:
    """Read official annotation file, resolve photo + sketch paths. Returns (pairs, problems)."""
    anno = json.loads((root / f"anno_{split}.json").read_text())
    pairs, problems = [], []
    for a in anno:
        name = a["image_name"]                                   # "photo1/image0110"
        folder, stem = name.split("/")
        k = folder.replace("photo", "")
        photo = _find(root / "photo" / folder / stem)
        sketch = _find(root / "sketch" / f"sketch{k}" / stem.replace("image", "sketch"))
        if photo is None or sketch is None:
            problems.append(f"{name}: photo={photo} sketch={sketch}")
            continue
        style = int(a["style"])
        if style not in (0, 1, 2):
            problems.append(f"{name}: unexpected style {style}")
            continue
        pairs.append({"name": name, "photo": str(photo.relative_to(root)),
                      "sketch": str(sketch.relative_to(root)), "style": style})
    return pairs, problems


def build_cache_and_split(root: Path = FS2K_DIR, seed: int = 42, val_frac: float = 0.15) -> dict:
    """Cache 128px photos/sketches as uint8 arrays; stratified-by-style 15% val split of official train."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "test"):
        pairs, problems = resolve_pairs(root, split)
        report[split] = {"pairs": len(pairs), "problems": problems[:50], "n_problems": len(problems)}
        photos = np.stack([load_rgb_128(root / p["photo"]) for p in pairs])
        sketches = np.stack([load_rgb_128(root / p["sketch"]) for p in pairs])
        np.save(CACHE_DIR / f"fs2k_{split}_photo_128.npy", photos)
        np.save(CACHE_DIR / f"fs2k_{split}_sketch_128.npy", sketches)
        save_json(pairs, CACHE_DIR / f"fs2k_{split}_pairs.json")
    train_pairs = load_json(CACHE_DIR / "fs2k_train_pairs.json")
    styles = [p["style"] for p in train_pairs]
    tr, va = train_test_split(np.arange(len(train_pairs)), test_size=val_frac, random_state=seed, stratify=styles)
    split = {"seed": seed, "val_frac": val_frac, "train_idx": sorted(tr.tolist()), "val_idx": sorted(va.tolist()),
             "style_counts": {s: {"train": int(sum(styles[i] == s for i in tr)),
                                  "val": int(sum(styles[i] == s for i in va))} for s in range(3)}}
    save_json(split, SPLIT_PATH)
    report["split"] = split["style_counts"]
    return report


def load_fs2k(subset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """subset in {'train','val','test'} -> (photos, sketches, styles)."""
    src = "test" if subset == "test" else "train"
    photos = np.load(CACHE_DIR / f"fs2k_{src}_photo_128.npy")
    sketches = np.load(CACHE_DIR / f"fs2k_{src}_sketch_128.npy")
    styles = np.array([p["style"] for p in load_json(CACHE_DIR / f"fs2k_{src}_pairs.json")])
    if subset != "test":
        idx = load_json(SPLIT_PATH)[f"{subset}_idx"]
        photos, sketches, styles = photos[idx], sketches[idx], styles[idx]
    return photos, sketches, styles


class PairedSketchDataset(Dataset):
    """Returns (photo [3,H,W] in [-1,1], sketch [3,H,W] in [-1,1], style int).

    Augmentation (train only) uses ONE set of random parameters for both images of a pair, so the
    pixel-level correspondence is preserved: horizontal flip + random-resized jitter (pix2pix style).
    """

    def __init__(self, photos: np.ndarray, sketches: np.ndarray, styles: np.ndarray, augment: bool = False,
                 jitter: int = 8):
        self.photos, self.sketches, self.styles = photos, sketches, styles
        self.augment, self.jitter = augment, jitter

    def __len__(self) -> int:
        return len(self.photos)

    def _paired_aug(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
        import cv2
        s = a.shape[0]
        if rng.random() < 0.5:                                    # same flip for both
            a, b = a[:, ::-1], b[:, ::-1]
        if self.jitter > 0:                                       # same upscale + crop window for both
            big = s + self.jitter
            a = cv2.resize(np.ascontiguousarray(a), (big, big), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(np.ascontiguousarray(b), (big, big), interpolation=cv2.INTER_LINEAR)
            x, y = int(rng.integers(0, self.jitter + 1)), int(rng.integers(0, self.jitter + 1))
            a, b = a[y:y + s, x:x + s], b[y:y + s, x:x + s]
        return a, b

    def __getitem__(self, i):
        a, b = self.photos[i], self.sketches[i]
        if self.augment:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            a, b = self._paired_aug(a, b, rng)
        return to_tensor(a) * 2 - 1, to_tensor(b) * 2 - 1, int(self.styles[i])
