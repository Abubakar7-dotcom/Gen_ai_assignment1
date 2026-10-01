"""Deterministic validation and test corruption manifests.

val : one corruption per validation image, conditions balanced (25% each), params from the TRAINING distribution.
test: every test image x {clean} + {salt_pepper, blur, occlusion} x {low, medium, high} = 10 items per image.
Each item stores: index (row in the cached array), id, condition, severity, params (incl. rects / blur / p), seed.
"""
from __future__ import annotations

import numpy as np

from src.data.corruptions import CONDITIONS, SEVERITIES, sample_params, severity_of, severity_params

MANIFEST_SEED = 42


def build_val_manifest(val_ids: list[str], seed: int = MANIFEST_SEED, size: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    conds = np.resize(np.arange(len(CONDITIONS)), len(val_ids))
    rng.shuffle(conds)
    items = []
    for i, (img_id, c) in enumerate(zip(val_ids, conds)):
        item_seed = int(rng.integers(0, 2**31 - 1))
        cond = CONDITIONS[int(c)]
        params = sample_params(cond, np.random.default_rng(item_seed), size=size)
        items.append({"index": i, "id": img_id, "condition": cond,
                      "severity": severity_of(cond, params), "params": params, "seed": item_seed})
    return {"split": "val", "seed": seed, "items": items}


def build_test_manifest(test_ids: list[str], seed: int = MANIFEST_SEED + 1, size: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    items = []
    for i, img_id in enumerate(test_ids):
        items.append({"index": i, "id": img_id, "condition": "clean", "severity": "none", "params": {},
                      "seed": None})
        for cond in CONDITIONS[1:]:
            for sev in SEVERITIES:
                item_seed = int(rng.integers(0, 2**31 - 1))
                params = severity_params(cond, sev, np.random.default_rng(item_seed), size=size)
                items.append({"index": i, "id": img_id, "condition": cond, "severity": sev,
                              "params": params, "seed": item_seed})
    return {"split": "test", "seed": seed, "items": items}
