"""Lazy ONNX Runtime session registry. Missing model files produce a clear 503 instead of a crash."""
from __future__ import annotations

import threading
from pathlib import Path

import onnxruntime as ort
from fastapi import HTTPException

MODEL_FILES = ["t1_universal", "t2_classifier", "t2_spec_salt_pepper", "t2_spec_blur", "t2_spec_occlusion",
               "t3_moe", "t4_generator"]


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        self.providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._opts = opts

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.onnx"

    def session(self, name: str) -> ort.InferenceSession:
        with self._lock:
            if name not in self._sessions:
                p = self.path(name)
                if not p.exists():
                    raise HTTPException(503, f"model {p.name} not found in {self.dir}. "
                                             f"Run scripts/download_models to fetch the ONNX files.")
                self._sessions[name] = ort.InferenceSession(str(p), self._opts, providers=self.providers)
            return self._sessions[name]

    def run(self, name: str, feeds: dict):
        return self.session(name).run(None, feeds)

    def status(self) -> dict:
        return {n: {"available": self.path(n).exists(), "loaded": n in self._sessions} for n in MODEL_FILES}
