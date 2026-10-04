"""Make sure the ONNX models are present before the API starts (runs as the container's first step).

For every model in MODEL_FILES: keep it if its SHA-256 matches the release checksum list, otherwise download it
from the GitHub Release (MODEL_BASE_URL). Downloads go to a .part file and are renamed only after the checksum
matches, so an interrupted download never leaves a broken model behind. Without internet access the API still
starts; missing models then return a clear 503 and /health shows which ones are absent.

    python -m app.fetch_models          # MODELS_DIR and MODEL_BASE_URL come from the environment
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from app.models import MODEL_FILES

BASE_URL = os.environ.get("MODEL_BASE_URL",
                          "https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/releases/download/models-v1")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "models_onnx"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hashes() -> dict[str, str]:
    with urllib.request.urlopen(f"{BASE_URL}/SHA256SUMS.txt", timeout=60) as r:
        lines = r.read().decode().splitlines()
    return {name: digest for digest, name in (ln.split() for ln in lines if ln.strip())}


def download(name: str, digest: str) -> None:
    part = MODELS_DIR / f"{name}.part"
    print(f"  downloading {name} ...", flush=True)
    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if sha256(part) != digest:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {name}")
    os.replace(part, MODELS_DIR / name)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        hashes = expected_hashes()
    except OSError as e:                                   # offline: keep whatever is there
        missing = [m for m in MODEL_FILES if not (MODELS_DIR / f"{m}.onnx").exists()]
        print(f"fetch_models: cannot reach {BASE_URL} ({e}); missing models: {missing or 'none'}", flush=True)
        return 0
    for m in MODEL_FILES:
        name = f"{m}.onnx"
        path = MODELS_DIR / name
        if path.exists() and sha256(path) == hashes[name]:
            print(f"  ok   {name}", flush=True)
            continue
        try:
            download(name, hashes[name])
        except (OSError, RuntimeError) as e:
            print(f"  FAILED {name}: {e}", flush=True)
    print("fetch_models: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
