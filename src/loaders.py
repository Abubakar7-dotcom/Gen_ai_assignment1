"""Load trained models from checkpoints (used by evaluation and ONNX export)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.autoencoder import DenoisingAE
from src.models.classifier import CorruptionClassifier
from src.models.moe import SoftMoE
from src.models.pix2pix import StyleUNetGenerator
from src.utils import ROOT, load_checkpoint

CKPT = {
    "t1": "checkpoints/t1/best.pt",
    "t2_cls": "checkpoints/t2_cls/best.pt",
    "t2_spec_salt_pepper": "checkpoints/t2_spec_salt_pepper/best.pt",
    "t2_spec_blur": "checkpoints/t2_spec_blur/best.pt",
    "t2_spec_occlusion": "checkpoints/t2_spec_occlusion/best.pt",
    "t3": "checkpoints/t3/best.pt",
    "t4": "checkpoints/t4/best_G.pt",
}
SPECIALISTS = ["salt_pepper", "blur", "occlusion"]


def _cls_from_cfg(cfg: dict) -> CorruptionClassifier:
    return CorruptionClassifier(channels=tuple(cfg["channels"]), dropout=cfg["dropout"], n_classes=cfg["n_classes"])


def load_ae(key: str) -> DenoisingAE:
    ck = load_checkpoint(ROOT / CKPT[key])
    m = DenoisingAE(**ck["model_cfg"]); m.load_state_dict(ck["model"])
    return m.eval()


def load_classifier() -> CorruptionClassifier:
    ck = load_checkpoint(ROOT / CKPT["t2_cls"])
    m = _cls_from_cfg(ck["model_cfg"]); m.load_state_dict(ck["model"])
    return m.eval()


def load_specialists() -> dict[str, DenoisingAE]:
    return {s: load_ae(f"t2_spec_{s}") for s in SPECIALISTS}


def load_moe() -> SoftMoE:
    ck = load_checkpoint(ROOT / CKPT["t3"])
    moe = SoftMoE(_cls_from_cfg(ck["gate_cfg"]), [DenoisingAE(**c) for c in ck["expert_cfgs"]], tau=ck["tau"])
    moe.load_state_dict(ck["moe"])
    return moe.eval()


def load_generator() -> StyleUNetGenerator:
    ck = load_checkpoint(ROOT / CKPT["t4"])
    g = StyleUNetGenerator(**ck["g_cfg"]); g.load_state_dict(ck["G"])
    return g.eval()


class HardRouter(nn.Module):
    """Task-2 inference: argmax classifier -> identity bypass for clean, else the chosen specialist."""

    def __init__(self, classifier: nn.Module, specialists: dict[str, nn.Module]):
        super().__init__()
        self.classifier = classifier
        self.specialists = nn.ModuleDict(specialists)

    @torch.no_grad()
    def forward(self, x, route: torch.Tensor | None = None):
        probs = torch.softmax(self.classifier(x), 1)
        r = probs.argmax(1) if route is None else route            # oracle routing passes the true label
        out = x.clone()
        for k, name in enumerate(SPECIALISTS, start=1):
            m = r == k
            if m.any():
                out[m] = self.specialists[name](x[m])
        return out, probs, r
