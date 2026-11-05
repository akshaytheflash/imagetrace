"""Datasets for the fusion model.

Two sources, one interface:

* procedural scenes (default) - infinite, perfectly labelled, useful for a
  cold start and for CI;
* a folder of real photographs - used as host and donor material, with the
  same edit simulator applied. Point ``--data`` at any image folder.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..forensics.stack import build_features
from ..io_utils import bundle_from_array
from ..synth import make_tampered, render_scene

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class TamperDataset(Dataset):
    def __init__(self, length: int = 512, size: tuple[int, int] = (256, 256),
                 n_edits: tuple[int, int] = (1, 3), seed_offset: int = 0,
                 folder: str | Path | None = None, clean_prob: float = 0.15):
        self.length = length
        self.size = size
        self.n_edits = n_edits
        self.seed_offset = seed_offset
        self.clean_prob = clean_prob
        self.paths: list[Path] = []
        self._cache: dict[int, np.ndarray] = {}
        if folder:
            root = Path(folder)
            self.paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT]
            if not self.paths:
                raise FileNotFoundError(f"no images under {folder}")
        else:
            # Render a pool once: fractal noise synthesis is the expensive part
            # of the generator, not the editing.
            self.pool = [render_scene(700_000 + i, size=(320, 320))
                         for i in range(24)]

    def __len__(self) -> int:
        return self.length

    def _real(self, idx: int) -> np.ndarray:
        key = idx % len(self.paths)
        if key not in self._cache:
            img = cv2.imread(str(self.paths[key]), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((*self.size, 3), np.uint8)
            self._cache[key] = img
        return self._cache[key]

    def __getitem__(self, idx: int):
        seed = self.seed_offset + idx
        h, w = self.size
        host = donor = None
        if self.paths:
            host = self._real(seed * 2 + 1)
            donor = self._real(seed * 2 + 7)
        elif getattr(self, "pool", None):
            rng = np.random.default_rng(seed + 5)
            ha, ho = rng.integers(0, len(self.pool), 2)
            def crop(img):
                ch, cw = img.shape[:2]
                if ch <= h or cw <= w:
                    return img
                y0 = int(rng.integers(0, ch - h + 1))
                x0 = int(rng.integers(0, cw - w + 1))
                return img[y0:y0 + h, x0:x0 + w]
            host = crop(self.pool[int(ha)])
            donor = crop(self.pool[int(ho)])
        n = int(np.random.default_rng(seed).integers(self.n_edits[0], self.n_edits[1] + 1))
        if np.random.default_rng(seed + 991).random() < self.clean_prob:
            n = 0
        sample = make_tampered(seed, size=(h, w), n_edits=n, host=host, donor=donor)
        stack = build_features(bundle_from_array(sample.bgr))
        x = torch.from_numpy(stack.tensor())
        y = torch.from_numpy((sample.mask > 0).astype(np.float32))[None]
        return x, y


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)
