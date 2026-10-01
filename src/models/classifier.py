"""Corruption classifier (Task 2) — also the initialisation of the Task-3 gating network.

Predicts 4 classes: 0=clean, 1=salt_pepper, 2=blur, 3=occlusion. Returns LOGITS; softmax is applied outside.
The first stage keeps full resolution because blur vs. clean is a high-frequency cue that early pooling destroys.
"""
from __future__ import annotations

import torch.nn as nn


class CorruptionClassifier(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.2, n_classes: int = 4):
        super().__init__()
        self.cfg = dict(channels=list(channels), dropout=dropout, n_classes=n_classes)
        layers, cin = [], 3
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
