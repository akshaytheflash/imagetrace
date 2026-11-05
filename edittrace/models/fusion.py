"""Checkpoint loading and tiled inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import MODEL_CHANNELS, Config
from ..forensics.stack import FeatureStack
from .unet import UNet, pad_to


class FusionModel:
    def __init__(self, path: str | Path, device: str = "cpu", width: int = 24, depth: int = 4):
        self.device = torch.device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.width = int(ckpt.get("width", width)) if isinstance(ckpt, dict) else width
        self.depth = int(ckpt.get("depth", depth)) if isinstance(ckpt, dict) else depth
        in_ch = int(next(iter(state.values())).shape[1])
        self.model = UNet(in_ch=in_ch, width=self.width, depth=self.depth)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.meta = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}

    @property
    def channels(self) -> int:
        return int(next(iter(self.model.parameters())).shape[1])

    @torch.no_grad()
    def predict(self, stack: FeatureStack, tile: int = 640) -> np.ndarray:
        x = torch.from_numpy(stack.tensor()).unsqueeze(0).to(self.device)
        h, w = x.shape[-2:]
        if max(h, w) <= tile:
            xp, (hh, ww) = pad_to(x, 16)
            y = self.model(xp)[:, 0, :hh, :ww]
            return torch.sigmoid(y)[0].cpu().numpy().astype(np.float32)

        # Tiled inference with 64px overlap, averaged in the overlap region.
        out = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        step = tile - 64
        for y0 in range(0, max(1, h - 32), step):
            for x0 in range(0, max(1, w - 32), step):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                if y1 - y0 < 32 or x1 - x0 < 32:
                    continue
                crop = x[:, :, y0:y1, x0:x1]
                cp, (ch, cw) = pad_to(crop, 16)
                p = torch.sigmoid(self.model(cp)[:, 0, :ch, :cw])[0].cpu().numpy()
                out[y0:y1, x0:x1] += p
                weight[y0:y1, x0:x1] += 1.0
        weight[weight == 0] = 1.0
        return (out / weight).astype(np.float32)


def blend_scores(model_prob: np.ndarray, classical: np.ndarray,
                 weight: float = 0.62) -> np.ndarray:
    """Convex blend, then a mild contrast stretch so the mask threshold is
    meaningful even when the network is under-confident."""
    if weight >= 0.999:
        return np.clip(model_prob, 0.0, 1.0).astype(np.float32)
    p = weight * model_prob + (1.0 - weight) * classical
    return np.clip((p - 0.5) * 1.25 + 0.5, 0.0, 1.0).astype(np.float32)


def load_if_available(cfg: Config) -> FusionModel | None:
    path = cfg.resolved_model_path()
    if path is None:
        return None
    try:
        return FusionModel(path, device=cfg.device)
    except Exception:
        return None
