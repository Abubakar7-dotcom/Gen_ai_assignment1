"""FastAPI inference service for the four workspaces (ONNX Runtime, CPU).

Endpoints
  GET  /health                 service + model status
  GET  /samples                names of bundled clean sample images
  GET  /samples/{name}         one sample image (PNG)
  GET  /corruptions            available conditions, severities and their exact settings
  POST /corrupt                apply a corruption (same code as training) -> corrupted image + params
  POST /restore/universal      Task 1
  POST /restore/hard           Task 2 (classifier -> specialist / identity bypass)
  POST /restore/moe            Task 3 (soft mixture of experts)
  POST /sketch                 Task 4 (style-conditioned face-to-sketch)
Restoration endpoints accept either an uploaded image (``file``) or a bundled ``sample`` name, plus an optional
corruption to apply first (``condition`` + ``severity`` low|medium|high|random, optional ``seed``).
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.corruptions import (CONDITIONS, SEVERITIES, TEST_SEVERITY, apply_corruption, sample_params,
                             severity_params)
from app.models import ModelRegistry

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "models_onnx"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))
MAX_BYTES = 10 * 2**20
IMG = 128
EXPERT_NAMES = {"clean": "Identity bypass", "salt_pepper": "Salt-and-pepper expert", "blur": "Blur expert",
                "occlusion": "Occlusion expert"}

app = FastAPI(title="GenAI A1 — Restoration & Sketch API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
registry = ModelRegistry(MODELS_DIR)


# ----------------------------------------------------------------- image helpers
def b64png(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def decode_upload(data: bytes) -> np.ndarray:
    if len(data) == 0:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // 2**20} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "could not decode image (use PNG/JPEG/WebP/BMP)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Same as training: RGB, 128x128 (INTER_AREA)."""
    return cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)


async def get_image(file: UploadFile | None, sample: str | None) -> tuple[np.ndarray, str]:
    if file is not None and file.filename:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(415, f"expected an image, got {file.content_type}")
        return preprocess(decode_upload(await file.read())), file.filename
    if sample:
        p = (SAMPLES_DIR / Path(sample).name)
        if not p.exists():
            raise HTTPException(404, f"sample {sample!r} not found")
        return preprocess(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)), p.name
    raise HTTPException(400, "provide an image file or a sample name")


def maybe_corrupt(img: np.ndarray, condition: str | None, severity: str | None, seed: int | None):
    if not condition or condition == "none":
        return img, None
    if condition not in CONDITIONS:
        raise HTTPException(422, f"condition must be one of {CONDITIONS}")
    rng = np.random.default_rng(seed)
    if condition == "clean":
        params = {}
    elif severity in SEVERITIES:
        params = severity_params(condition, severity, rng)
    elif severity in (None, "", "random"):
        params = sample_params(condition, rng)
    else:
        raise HTTPException(422, f"severity must be one of {SEVERITIES + ['random']}")
    return apply_corruption(img, condition, params), {"condition": condition, "severity": severity or "random",
                                                      "params": params, "seed": seed}


def to_input(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def from_output(t: np.ndarray) -> np.ndarray:
    return (np.clip(t[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "models": registry.status(), "models_dir": str(MODELS_DIR),
            "onnxruntime_providers": registry.providers}


@app.get("/samples")
def samples():
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    return {"samples": sorted(p.name for p in SAMPLES_DIR.iterdir()
                              if p.suffix.lower() in (".png", ".jpg", ".jpeg"))}


@app.get("/samples/{name}")
def sample_image(name: str):
    p = SAMPLES_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404, "not found")
    return Response(p.read_bytes(), media_type="image/png" if p.suffix == ".png" else "image/jpeg")


@app.get("/corruptions")
def corruptions():
    return {"conditions": CONDITIONS, "severities": SEVERITIES + ["random"], "test_severity": TEST_SEVERITY,
            "training_ranges": {"salt_pepper": "p ~ U(0.02, 0.15)", "blur": "k in {3,5,7}, sigma ~ U(0.5, 2.5)",
                                "occlusion": "1-3 black rectangles, union area 10-35%"}}


@app.post("/corrupt")
async def corrupt(file: UploadFile | None = File(None), sample: str | None = Form(None),
                  condition: str = Form(...), severity: str | None = Form("medium"), seed: int | None = Form(None)):
    img, src = await get_image(file, sample)
    out, info = maybe_corrupt(img, condition, severity, seed)
    return {"source": src, "clean": b64png(img), "corrupted": b64png(out), "corruption": info}


@app.post("/restore/universal")
async def restore_universal(file: UploadFile | None = File(None), sample: str | None = Form(None),
                            condition: str | None = Form(None), severity: str | None = Form(None),
                            seed: int | None = Form(None)):
    img, src = await get_image(file, sample)
    x, info = maybe_corrupt(img, condition, severity, seed)
    t0 = time.perf_counter()
    out = registry.run("t1_universal", {"input": to_input(x)})[0]
    ms = (time.perf_counter() - t0) * 1000
    return {"source": src, "input": b64png(x), "output": b64png(from_output(out)), "corruption": info,
            "inference_ms": round(ms, 2), "model": "t1_universal.onnx"}


@app.post("/restore/hard")
async def restore_hard(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       condition: str | None = Form(None), severity: str | None = Form(None),
                       seed: int | None = Form(None)):
    img, src = await get_image(file, sample)
    x, info = maybe_corrupt(img, condition, severity, seed)
    inp = to_input(x)
    t0 = time.perf_counter()
    probs = registry.run("t2_classifier", {"input": inp})[0][0]
    t_cls = time.perf_counter()
    pred = CONDITIONS[int(probs.argmax())]
    if pred == "clean":                                  # identity bypass: no restoration expert is run
        out = x
    else:
        out = from_output(registry.run(f"t2_spec_{pred}", {"input": inp})[0])
    t1 = time.perf_counter()
    return {"source": src, "input": b64png(x), "output": b64png(out), "corruption": info,
            "probabilities": {c: round(float(p), 5) for c, p in zip(CONDITIONS, probs)},
            "predicted": pred, "expert": EXPERT_NAMES[pred],
            "inference_ms": round((t1 - t0) * 1000, 2), "classifier_ms": round((t_cls - t0) * 1000, 2)}


@app.post("/restore/moe")
async def restore_moe(file: UploadFile | None = File(None), sample: str | None = Form(None),
                      condition: str | None = Form(None), severity: str | None = Form(None),
                      seed: int | None = Form(None)):
    img, src = await get_image(file, sample)
    x, info = maybe_corrupt(img, condition, severity, seed)
    t0 = time.perf_counter()
    out, w = registry.run("t3_moe", {"input": to_input(x)})
    ms = (time.perf_counter() - t0) * 1000
    weights = {c: round(float(v), 5) for c, v in zip(CONDITIONS, w[0])}
    ranked = sorted(weights, key=weights.get, reverse=True)
    return {"source": src, "input": b64png(x), "output": b64png(from_output(out)), "corruption": info,
            "weights": weights, "top_expert": EXPERT_NAMES[ranked[0]],
            "ranking": [EXPERT_NAMES[k] for k in ranked], "inference_ms": round(ms, 2)}


@app.post("/sketch")
async def sketch(file: UploadFile | None = File(None), sample: str | None = Form(None), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3")
    img, src = await get_image(file, sample)
    photo = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    t0 = time.perf_counter()
    out = registry.run("t4_generator", {"photo": photo, "style": np.array([style - 1], dtype=np.int64)})[0]
    ms = (time.perf_counter() - t0) * 1000
    sk = ((np.clip(out[0].transpose(1, 2, 0), -1, 1) + 1) * 127.5).round().astype(np.uint8)
    return {"source": src, "photo": b64png(img), "sketch": b64png(sk), "style": style, "inference_ms": round(ms, 2)}
