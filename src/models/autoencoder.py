"""Convolutional denoising autoencoder used by Task 1 (universal) and Task 2 (specialists).

    x~ [3,128,128] --E--> z [latent_ch, 8, 8] --D--> x^ [3,128,128]

Encoder: 4 stride-2 stages (128->64->32->16->8), channels base*(1,2,4,8).
Bottleneck: 1x1 conv down to ``latent_ch`` channels at 8x8 = the compressed code. With latent_ch=16 the code
has 1,024 values vs 49,152 input values (48x compression), so the network cannot just copy the input.
Decoder mirrors the encoder with bilinear upsampling + conv (avoids transposed-conv checkerboard artefacts).

``skip`` (default False) enables ONE limited skip from the 32x32 encoder stage, squeezed to 8 channels.
It exists only as an ablation (see docs/decisions.md); the submitted Task-1 model uses skip=False.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
    )


class Encoder(nn.Module):
    def __init__(self, base: int = 32, latent_ch: int = 16, dropout: float = 0.0):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.stem = conv_block(3, base)                                       # 128
        self.down = nn.ModuleList([conv_block(ch[max(i - 1, 0)] if i else base, ch[i], stride=2)
                                   for i in range(4)])                        # 64, 32, 16, 8
        self.to_latent = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(ch[-1], latent_ch, 1))

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for blk in self.down:
            h = blk(h)
            feats.append(h)
        return self.to_latent(h), feats


class Decoder(nn.Module):
    def __init__(self, base: int = 32, latent_ch: int = 16, skip: bool = False):
        super().__init__()
        ch = [base * 8, base * 4, base * 2, base]
        self.skip = skip
        self.from_latent = conv_block(latent_ch, ch[0])                       # 8
        self.up = nn.ModuleList()
        cin = ch[0]
        for i, cout in enumerate(ch[1:] + [base]):                            # 16, 32, 64, 128
            extra = 8 if (skip and i == 1) else 0                             # skip enters at 32x32
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                         conv_block(cin + extra, cout)))
            cin = cout
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, z, skip_feat=None):
        h = self.from_latent(z)
        for i, blk in enumerate(self.up):
            if self.skip and i == 1:
                h = blk[0](h)
                h = blk[1](torch.cat([h, skip_feat], 1))
            else:
                h = blk(h)
        return torch.sigmoid(self.head(h))


class DenoisingAE(nn.Module):
    def __init__(self, base: int = 32, latent_ch: int = 16, dropout: float = 0.0, skip: bool = False):
        super().__init__()
        self.cfg = dict(base=base, latent_ch=latent_ch, dropout=dropout, skip=skip)
        self.encoder = Encoder(base, latent_ch, dropout)
        self.decoder = Decoder(base, latent_ch, skip)
        self.skip_proj = nn.Conv2d(base * 2, 8, 1) if skip else None          # 32x32 feature -> 8 channels

    def forward(self, x):
        z, feats = self.encoder(x)
        sk = self.skip_proj(feats[1]) if self.skip_proj is not None else None
        return self.decoder(z, sk)

    def latent_size(self, img: int = 128) -> int:
        return self.cfg["latent_ch"] * (img // 16) ** 2
