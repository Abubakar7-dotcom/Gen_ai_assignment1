"""Oxford-IIIT Pet: download, 128x128 cache, seed-42 80/20 split, runtime-corrupted and manifest datasets.

Layout expected under ``data/raw/oxford-iiit-pet`` (official archive layout):
    images/*.jpg
    annotations/trainval.txt, annotations/test.txt
Cache written to ``data/cache``:  pets_trainval_128.npy, pets_test_128.npy (uint8 N x 128 x 128 x 3) + *_ids.json
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.corruptions import COND_TO_IDX, CONDITIONS, apply_corruption, sample_params
from src.utils import ROOT, load_json, save_json

RAW_DIR = ROOT / "data" / "raw" / "oxford-iiit-pet"
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_PATH = ROOT / "data" / "splits" / "pets_split.json"
IMG_SIZE = 128
URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}


# ------------------------------------------------------------------ download / cache
def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if (raw_dir / name).exists():
            continue
        archive = raw_dir / f"{name}.tar.gz"
        if not archive.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as t:
            t.extractall(raw_dir)


def read_official_ids(raw_dir: Path, split: str) -> list[str]:
    """Image ids from annotations/{trainval,test}.txt (first column), in file order."""
    lines = (raw_dir / "annotations" / f"{split}.txt").read_text().splitlines()
    return [ln.split()[0] for ln in lines if ln.strip() and not ln.startswith("#")]


def load_rgb_128(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """Read any image as RGB (handles grayscale / RGBA / CMYK jpgs) and resize to size x size."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)          # forces 3-channel BGR
    if img is None:                                         # rare decode failure -> PIL fallback
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def build_cache(raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("trainval", "test"):
        out = cache_dir / f"pets_{split}_128.npy"
        if out.exists():
            continue
        ids = read_official_ids(raw_dir, split)
        arr = np.stack([load_rgb_128(raw_dir / "images" / f"{i}.jpg") for i in ids])
        np.save(out, arr)
        save_json(ids, cache_dir / f"pets_{split}_ids.json")
        print(f"cached {split}: {arr.shape}")


def make_split(trainval_ids: list[str], seed: int = 42, train_frac: float = 0.8) -> dict:
    """80/20 split of the official trainval list with seed 42. Stored as both ids and row indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trainval_ids))
    n_train = int(round(train_frac * len(trainval_ids)))
    tr, va = sorted(perm[:n_train].tolist()), sorted(perm[n_train:].tolist())
    return {"seed": seed, "train_frac": train_frac,
            "train_idx": tr, "val_idx": va,
            "train_ids": [trainval_ids[i] for i in tr], "val_ids": [trainval_ids[i] for i in va]}


def load_arrays(cache_dir: Path = CACHE_DIR) -> dict[str, np.ndarray]:
    """Returns {'train','val','test'} uint8 arrays using the saved split."""
    split = load_json(SPLIT_PATH)
    tv = np.load(cache_dir / "pets_trainval_128.npy", mmap_mode="r")
    te = np.load(cache_dir / "pets_test_128.npy", mmap_mode="r")
    return {"train": np.ascontiguousarray(tv[split["train_idx"]]),
            "val": np.ascontiguousarray(tv[split["val_idx"]]),
            "test": te}


# ------------------------------------------------------------------ tensors
def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float CHW in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """float CHW [0,1] -> uint8 HWC."""
    return (t.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())


# ------------------------------------------------------------------ datasets
class RuntimeCorruptionDataset(Dataset):
    """Training dataset: corruption type + severity resampled on EVERY load (nothing saved to disk).

    Index can be an int (condition sampled uniformly from ``conditions``) or an ``(idx, condition_idx)``
    tuple coming from :class:`BalancedConditionBatchSampler`.
    Returns (corrupted [3,H,W], clean [3,H,W], condition label int).
    """

    def __init__(self, images: np.ndarray, conditions: list[str] | None = None):
        self.images = images
        self.conditions = conditions or CONDITIONS

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            idx, cond = int(index[0]), CONDITIONS[int(index[1])]
        else:
            idx, cond = int(index), None
        # torch seeds each worker differently every epoch -> fresh, reproducible-per-run randomness
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        if cond is None:
            cond = self.conditions[int(rng.integers(len(self.conditions)))]
        clean = self.images[idx]
        corrupted = apply_corruption(clean, cond, sample_params(cond, rng, size=clean.shape[0]))
        return to_tensor(corrupted), to_tensor(clean), COND_TO_IDX[cond]


class ManifestDataset(Dataset):
    """Deterministic val/test dataset driven by a manifest (list of dicts with index/condition/params)."""

    def __init__(self, images: np.ndarray, items: list[dict], conditions: list[str] | None = None):
        self.images = images
        self.items = [it for it in items if conditions is None or it["condition"] in conditions]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        clean = self.images[it["index"]]
        corrupted = apply_corruption(clean, it["condition"], it["params"])
        return to_tensor(corrupted), to_tensor(clean), COND_TO_IDX[it["condition"]], i
