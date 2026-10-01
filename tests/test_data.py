import numpy as np
import pytest
import torch

from src.data.corruptions import (BLUR_KERNELS, CONDITIONS, SEVERITIES, TEST_SEVERITY, apply_corruption,
                                  sample_params, severity_params, union_area_fraction)
from src.data.manifests import build_test_manifest, build_val_manifest
from src.data.pets import ManifestDataset, RuntimeCorruptionDataset, make_split
from src.data.sampler import BalancedConditionBatchSampler


@pytest.fixture
def img():
    return (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)


def test_training_param_ranges():
    rng = np.random.default_rng(1)
    for _ in range(300):
        p = sample_params("salt_pepper", rng); assert 0.02 <= p["p"] <= 0.15
        b = sample_params("blur", rng); assert b["kernel"] in BLUR_KERNELS and 0.5 <= b["sigma"] <= 2.5
        o = sample_params("occlusion", rng)
        assert 1 <= o["n"] <= 3 and len(o["rects"]) == o["n"]
        assert 0.10 <= union_area_fraction((128, 128), o["rects"]) <= 0.35


def test_test_severities():
    rng = np.random.default_rng(2)
    for sev in SEVERITIES:
        o = severity_params("occlusion", sev, rng)
        spec = TEST_SEVERITY["occlusion"][sev]
        assert o["n"] == spec["n"] and abs(o["area"] - spec["area"]) <= 0.0151
        assert severity_params("salt_pepper", sev, rng)["p"] == TEST_SEVERITY["salt_pepper"][sev]["p"]


def test_deterministic_given_params(img):
    for c in CONDITIONS:
        p = sample_params(c, np.random.default_rng(3))
        assert np.array_equal(apply_corruption(img, c, p), apply_corruption(img, c, p))


def test_salt_pepper_values(img):
    out = apply_corruption(img, "salt_pepper", {"p": 0.15, "seed": 0})
    changed = np.any(out != img, axis=2)
    vals = out[changed]
    assert np.all((vals == 0) | (vals == 255))
    assert 0.10 < changed.mean() < 0.20


def test_occlusion_black(img):
    p = severity_params("occlusion", "high", np.random.default_rng(4))
    out = apply_corruption(img, "occlusion", p)
    x, y, w, h = p["rects"][0]
    assert out[y:y + h, x:x + w].max() == 0


def test_balanced_sampler():
    s = BalancedConditionBatchSampler(n_items=100, batch_size=16)
    for batch in s:
        counts = np.bincount([c for _, c in batch], minlength=4)
        assert np.all(counts == 4)


def test_datasets(img):
    imgs = np.stack([img] * 8)
    ds = RuntimeCorruptionDataset(imgs)
    x, y, c = ds[(0, 2)]
    assert x.shape == (3, 128, 128) and c == 2 and torch.all((x >= 0) & (x <= 1))
    man = build_val_manifest([f"id{i}" for i in range(8)])
    mds = ManifestDataset(imgs, man["items"])
    a1, _, _, _ = mds[3]; a2, _, _, _ = mds[3]
    assert torch.equal(a1, a2)


def test_split_and_manifests():
    ids = [f"img{i}" for i in range(100)]
    s1, s2 = make_split(ids), make_split(ids)
    assert s1 == s2 and len(s1["train_idx"]) == 80 and not set(s1["train_idx"]) & set(s1["val_idx"])
    t = build_test_manifest(ids[:5])
    assert len(t["items"]) == 50
    v = build_val_manifest(ids)
    assert np.all(np.bincount([CONDITIONS.index(i["condition"]) for i in v["items"]]) == 25)
