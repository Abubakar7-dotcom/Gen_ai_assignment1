"""Corruption pipeline: the ONLY implementation, shared by training, manifests, evaluation and the FastAPI backend.

All functions take / return uint8 RGB images of shape (H, W, 3). Every corruption is fully described by a
JSON-serialisable ``params`` dict, so applying the same params to the same image always gives the same output.

Spec (assignment):
    clean        : unchanged
    salt_pepper  : p ~ U(0.02, 0.15); chosen pixels -> black or white with prob 0.5 each
    blur         : kernel in {3,5,7}; sigma ~ U(0.5, 2.5)
    occlusion    : 1-3 black rectangles, random positions, jointly covering 10%-35% of the image (union area)
Test severities (low / medium / high):
    salt_pepper p        : 0.03 / 0.08 / 0.15
    blur (kernel, sigma) : (3,0.7) / (5,1.5) / (7,2.5)
    occlusion            : ~10% with 1 rect / ~20% with 2 rects / ~35% with 3 rects
"""
from __future__ import annotations

import cv2
import numpy as np

CONDITIONS = ["clean", "salt_pepper", "blur", "occlusion"]   # label index order everywhere (= MoE branch order)
COND_TO_IDX = {c: i for i, c in enumerate(CONDITIONS)}
SEVERITIES = ["low", "medium", "high"]

# training ranges
SP_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_AREA_RANGE = (0.10, 0.35)
OCC_N_RANGE = (1, 3)

# fixed test severities
TEST_SEVERITY = {
    "salt_pepper": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {"low": {"kernel": 3, "sigma": 0.7}, "medium": {"kernel": 5, "sigma": 1.5},
             "high": {"kernel": 7, "sigma": 2.5}},
    "occlusion": {"low": {"area": 0.10, "n": 1}, "medium": {"area": 0.20, "n": 2}, "high": {"area": 0.35, "n": 3}},
}
OCC_TEST_TOL = 0.015   # "approximately" -> union area within +-1.5 percentage points of target


# ============================================================== apply (deterministic given params)
def apply_corruption(img: np.ndarray, condition: str, params: dict | None = None) -> np.ndarray:
    """Apply ``condition`` with fully specified ``params``. Returns a new uint8 array."""
    _check_img(img)
    params = params or {}
    if condition == "clean":
        return img.copy()
    if condition == "salt_pepper":
        return _salt_pepper(img, params["p"], params["seed"])
    if condition == "blur":
        return _blur(img, params["kernel"], params["sigma"])
    if condition == "occlusion":
        return _occlude(img, params["rects"])
    raise ValueError(f"unknown condition {condition!r}")


def _check_img(img: np.ndarray) -> None:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {img.dtype} {img.shape}")


def _salt_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w, _ = img.shape
    out = img.copy()
    mask = rng.random((h, w)) < p                    # pixel-level selection (all channels together)
    salt = rng.random((h, w)) < 0.5                  # black vs white with equal probability
    out[mask & salt] = 255
    out[mask & ~salt] = 0
    return out


def _blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (int(kernel), int(kernel)), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


def _occlude(img: np.ndarray, rects: list[list[int]]) -> np.ndarray:
    out = img.copy()
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = 0
    return out


def occlusion_mask(shape: tuple[int, int], rects: list[list[int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for x, y, w, h in rects:
        m[y:y + h, x:x + w] = True
    return m


def union_area_fraction(shape: tuple[int, int], rects: list[list[int]]) -> float:
    return float(occlusion_mask(shape, rects).mean())


# ============================================================== sampling params
def sample_params(condition: str, rng: np.random.Generator, size: int = 128) -> dict:
    """Sample params from the TRAINING distribution."""
    if condition == "clean":
        return {}
    if condition == "salt_pepper":
        return {"p": float(rng.uniform(*SP_RANGE)), "seed": int(rng.integers(0, 2**31 - 1))}
    if condition == "blur":
        return {"kernel": int(rng.choice(BLUR_KERNELS)), "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE))}
    if condition == "occlusion":
        n = int(rng.integers(OCC_N_RANGE[0], OCC_N_RANGE[1] + 1))
        rects, area = _sample_rects(rng, n, OCC_AREA_RANGE[0], OCC_AREA_RANGE[1], size)
        return {"n": n, "rects": rects, "area": area}
    raise ValueError(f"unknown condition {condition!r}")


def severity_params(condition: str, severity: str, rng: np.random.Generator, size: int = 128) -> dict:
    """Params for a FIXED test severity level (random parts — noise pattern, rect positions — from rng)."""
    if condition == "clean":
        return {}
    spec = TEST_SEVERITY[condition][severity]
    if condition == "salt_pepper":
        return {"p": spec["p"], "seed": int(rng.integers(0, 2**31 - 1))}
    if condition == "blur":
        return {"kernel": spec["kernel"], "sigma": spec["sigma"]}
    if condition == "occlusion":
        lo, hi = spec["area"] - OCC_TEST_TOL, spec["area"] + OCC_TEST_TOL
        rects, area = _sample_rects(rng, spec["n"], lo, hi, size, target=spec["area"])
        return {"n": spec["n"], "rects": rects, "area": area}
    raise ValueError(f"unknown condition {condition!r}")


def _sample_rects(rng: np.random.Generator, n: int, lo: float, hi: float, size: int,
                  target: float | None = None, max_tries: int = 500) -> tuple[list[list[int]], float]:
    """Rejection-sample n rectangles whose UNION covers a fraction in [lo, hi] of a size x size image.

    Overlap is allowed but measured: coverage is the union mask area, never the sum of rectangle areas.
    """
    total = size * size
    for _ in range(max_tries):
        a = target if target is not None else rng.uniform(lo, hi)
        shares = rng.dirichlet(np.ones(n)) if n > 1 else np.array([1.0])
        rects = []
        for s in shares:
            rect_area = max(16.0, a * s * total)
            aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))   # w/h in [0.5, 2]
            w = int(round(np.sqrt(rect_area * aspect)))
            h = int(round(rect_area / max(w, 1)))
            w, h = int(np.clip(w, 4, size)), int(np.clip(h, 4, size))
            x = int(rng.integers(0, size - w + 1))
            y = int(rng.integers(0, size - h + 1))
            rects.append([x, y, w, h])
        frac = union_area_fraction((size, size), rects)
        if lo <= frac <= hi:
            return rects, frac
    raise RuntimeError(f"could not sample {n} rects with union area in [{lo:.3f},{hi:.3f}]")


# ============================================================== convenience
def random_corruption(img: np.ndarray, rng: np.random.Generator, conditions: list[str] | None = None,
                      condition: str | None = None) -> tuple[np.ndarray, str, dict]:
    """Training-time corruption: pick a condition uniformly (or use the given one), sample params, apply."""
    if condition is None:
        conditions = conditions or CONDITIONS
        condition = conditions[int(rng.integers(len(conditions)))]
    params = sample_params(condition, rng, size=img.shape[0])
    return apply_corruption(img, condition, params), condition, params


def severity_of(condition: str, params: dict) -> str:
    """Bin a TRAINING-distribution param set into low/medium/high (equal thirds of the spec range).

    Used for validation-set reporting; the test set uses the fixed severities directly.
    """
    if condition == "clean":
        return "none"
    if condition == "salt_pepper":
        v, (a, b) = params["p"], SP_RANGE
    elif condition == "blur":
        v, (a, b) = params["sigma"], BLUR_SIGMA_RANGE
    elif condition == "occlusion":
        v, (a, b) = params["area"], OCC_AREA_RANGE
    else:
        raise ValueError(condition)
    t = (v - a) / (b - a)
    return "low" if t < 1 / 3 else ("medium" if t < 2 / 3 else "high")
