"""Style-conditioned pix2pix (Task 4): U-Net generator + PatchGAN discriminator.

Style condition s in {0,1,2} -> learned nn.Embedding(3, d). It is injected into
  * the generator twice: tiled as d extra input channels AND concatenated at the 1x1 bottleneck,
  * the discriminator: tiled as d extra input channels next to (photo, sketch).
Images are in [-1, 1]; generator ends with tanh.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _tile(e: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return e[:, :, None, None].expand(-1, -1, h, w)


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(cout, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.InstanceNorm2d(cout, affine=True),
                  nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.f = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.f(x), skip], 1)


class StyleUNetGenerator(nn.Module):
    """128x128 U-Net with 7 downsamplings (128 -> 1)."""

    def __init__(self, base: int = 64, style_dim: int = 16, n_styles: int = 3, dropout: float = 0.5):
        super().__init__()
        self.cfg = dict(base=base, style_dim=style_dim, n_styles=n_styles, dropout=dropout)
        b = base
        self.embed = nn.Embedding(n_styles, style_dim)
        ch = [b, b * 2, b * 4, b * 8, b * 8, b * 8, b * 8]                         # 64,32,16,8,4,2,1
        self.downs = nn.ModuleList()
        cin = 3 + style_dim
        for i, c in enumerate(ch):
            self.downs.append(Down(cin, c, norm=(0 < i < len(ch) - 1)))
            cin = c
        # bottleneck (1x1) receives the style embedding again
        self.ups = nn.ModuleList()
        up_out = [b * 8, b * 8, b * 8, b * 4, b * 2, b]                            # 2,4,8,16,32,64
        cin = ch[-1] + style_dim
        for i, c in enumerate(up_out):
            self.ups.append(Up(cin, c, dropout if i < 3 else 0.0))
            cin = c + ch[-2 - i]                                                   # concat with skip
        self.final = nn.Sequential(nn.ConvTranspose2d(cin, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.embed(style)
        h = torch.cat([x, _tile(e, x.shape[2], x.shape[3])], 1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = torch.cat([h, _tile(e, h.shape[2], h.shape[3])], 1)
        for i, u in enumerate(self.ups):
            h = u(h, skips[-2 - i])
        return self.final(h)


class StylePatchDiscriminator(nn.Module):
    """70x70-style PatchGAN on concat(photo, sketch, tiled style). Output: [B,1,14,14] logits for 128px input."""

    def __init__(self, base: int = 64, style_dim: int = 16, n_styles: int = 3):
        super().__init__()
        self.embed = nn.Embedding(n_styles, style_dim)
        b = base
        self.net = nn.Sequential(
            nn.Conv2d(6 + style_dim, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(b, b * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(b * 2, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(b * 2, b * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(b * 4, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(b * 4, b * 8, 4, 1, 1, bias=False), nn.InstanceNorm2d(b * 8, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(b * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        e = _tile(self.embed(style), photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_weights(m: nn.Module) -> None:
    """pix2pix init: N(0, 0.02) for conv weights."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
