"""Soft mixture-of-experts restoration (Task 3).

    w  = softmax(G(x~) / tau)                         # 4 weights: identity, salt, blur, occlusion
    x^ = w0 * x~ + w1 * A_salt(x~) + w2 * A_blur(x~) + w3 * A_occ(x~)

The gate is initialised from the Task-2 classifier and the experts from the Task-2 specialists.
``forward`` returns (x^, w, logits) so training can use CE on the gate; ``ExportMoE`` returns (x^, w) for ONNX.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SoftMoE(nn.Module):
    def __init__(self, gate: nn.Module, experts: list[nn.Module], tau: float = 1.0):
        super().__init__()
        assert len(experts) == 3, "experts must be [salt_pepper, blur, occlusion]"
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.tau = tau

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)                                # [B,4]
        outs = torch.stack([x] + [e(x) for e in self.experts], dim=1)              # [B,4,3,H,W]
        x_hat = (w[:, :, None, None, None] * outs).sum(1)
        return x_hat, w, logits

    def set_experts_trainable(self, flag: bool) -> None:
        for p in self.experts.parameters():
            p.requires_grad = flag
        # frozen experts also keep BatchNorm statistics fixed
        self.experts.train(flag and self.training)


class ExportMoE(nn.Module):
    """ONNX wrapper: one graph, outputs restored image and the 4 routing weights."""

    def __init__(self, moe: SoftMoE):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        x_hat, w, _ = self.moe(x)
        return x_hat, w
