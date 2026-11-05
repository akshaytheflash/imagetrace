"""A compact U-Net for fusing forensic channels into a manipulation mask.

Small on purpose: the forensic channels already do the heavy lifting, the
network only has to learn their spatial interaction. ~1.8M parameters trains
on CPU in minutes and keeps inference cheap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 18, width: int = 24, depth: int = 4):
        super().__init__()
        self.width = width
        self.depth = depth
        widths = [width * (2 ** min(i, 2)) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.enc.append(ConvBlock(prev, w))
            prev = w
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.Conv2d(widths[i + 1], widths[i], 1))
            self.dec.append(ConvBlock(widths[i] * 2, widths[i]))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < self.depth:
                feats.append(x)
                x = self.pool(x)
        for i in range(self.depth):
            skip = feats[self.depth - 1 - i]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skip], dim=1))
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_to(x: torch.Tensor, m: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)
