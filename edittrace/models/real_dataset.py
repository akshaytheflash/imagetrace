"""Supervised dataset over real tampered pairs (image + ground-truth mask)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..forensics.stack import build_features
from ..io_utils import bundle_from_array, encode_jpeg


class PairedMaskDataset(Dataset):
    """Rows are dicts with ``image``, optional ``mask`` and ``tampered``.

    Crops are taken after resizing the short side to 1.15x the target, with
    flips for augmentation and an optional JPEG round-trip so the model does
    not overfit to lossless PNG statistics.
    """

    def __init__(self, rows: list[dict], size: tuple[int, int] = (320, 320),
                 train: bool = True, jpeg_aug: float = 0.5):
        self.rows = rows
        self.size = size
        self.train = train
        self.jpeg_aug = jpeg_aug

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx * 131 + 7)
        row = self.rows[idx]
        h, w = self.size
        img = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros(img.shape[:2], np.uint8)
        if row.get("mask") is not None:
            m = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = cv2.resize(m, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        scale = 1.15
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if self.train:
            ys, xs = np.where(mask > 127)
            if ys.size > 0:
                # centre the crop on a tampered pixel so positives are never
                # missed; this is what keeps the model from collapsing to zero
                k = int(rng.integers(0, ys.size))
                cy, cx = int(ys[k]), int(xs[k])
                y0 = int(np.clip(cy - h // 2 + rng.integers(-h // 5, h // 5 + 1), 0, nh - h))
                x0 = int(np.clip(cx - w // 2 + rng.integers(-w // 5, w // 5 + 1), 0, nw - w))
            else:
                x0 = int(rng.integers(0, max(1, nw - w + 1)))
                y0 = int(rng.integers(0, max(1, nh - h + 1)))
            if rng.random() < 0.5:
                img = img[:, ::-1]
                mask = mask[:, ::-1]
                x0 = max(0, nw - w - x0)
        else:
            x0, y0 = (nw - w) // 2, (nh - h) // 2
        img = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
        mask = np.ascontiguousarray(mask[y0:y0 + h, x0:x0 + w])

        if self.train and self.jpeg_aug > 0 and rng.random() < self.jpeg_aug:
            q = int(rng.integers(70, 96))
            img = encode_jpeg(img, q)

        stack = build_features(bundle_from_array(img))
        x = torch.from_numpy(stack.tensor())
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, y
