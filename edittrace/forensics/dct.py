"""JPEG-specific forensics: ghost maps, block grid, double quantisation.

The JPEG pipeline is the most useful provenance recorder in consumer
photography. Every re-save imposes its quantisation lattice on 8x8 blocks; a
region pasted in after a save carries a *shorter* lattice history than its
surroundings. We expose three measurements:

* ``ghost_maps``   - recompress across a quality sweep and record the minimum
  and spread of the local residual (Farid's JPEG ghosts).
* ``blockiness_map`` - strength of the 8x8 grid, which breaks when a crop is
  pasted at a non-aligned offset or resized.
* ``dq_periodicity`` - period of the coefficient histogram, the fingerprint of
  double quantisation with an earlier quality factor.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

from ..io_utils import box_mean, robust_z


def ghost_maps(
    bgr: np.ndarray,
    qualities: tuple[int, ...] = (55, 60, 65, 70, 75, 80, 85, 90, 95),
    win: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ghost_min, ghost_spread, best_quality)``.

    ``ghost_min`` is low where the local content is well explained by *some*
    quality in the sweep. A tampered region typically reaches its minimum at a
    quality different from the rest of the image, which is what
    ``ghost_spread`` and ``best_quality`` record.
    """
    base = bgr.astype(np.float32)
    stack = np.empty((len(qualities),) + bgr.shape[:2], np.float32)
    for i, q in enumerate(qualities):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        rec = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) if ok else base
        diff = np.abs(base - rec).max(axis=2)
        stack[i] = box_mean(diff, win)

    idx = np.argmin(stack, axis=0)
    gmin = np.take_along_axis(stack, idx[None], axis=0)[0]
    gmax = stack.max(axis=0)
    best_q = np.asarray(qualities, np.float32)[idx]
    return gmin, (gmax - gmin), best_q


def blockiness_map(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ratio, z)`` for the 8x8 block grid at block resolution.

    The ratio compares gradient energy *on* block boundaries to energy inside
    blocks. Values > 1 indicate a strong JPEG grid; uniform interiors make it
    unstable, so the z-map is what downstream code consumes.
    """
    g = np.clip(gray * 255.0, 0, 255).astype(np.float32)
    h8, w8 = g.shape[0] // 8, g.shape[1] // 8
    if h8 < 2 or w8 < 2:
        z = np.zeros(g.shape, np.float32)
        return np.ones((max(h8, 1), max(w8, 1)), np.float32), z
    g = g[: h8 * 8, : w8 * 8]

    gx = np.abs(np.diff(g, axis=1))          # (H, W-1)
    gy = np.abs(np.diff(g, axis=0))          # (H-1, W)
    # Pad so the arrays align to whole 8x8 blocks; the padded line is the
    # image border and contributes no boundary evidence.
    gx = np.pad(gx, ((0, 0), (0, 1)))
    gy = np.pad(gy, ((0, 1), (0, 0)))

    gx = gx.reshape(h8, 8, w8, 8)            # block-row, y-offset, block-col, x-offset
    bx = gx[:, :, :, 7].mean(axis=1)         # boundary between block columns
    ix = gx[:, :, :, 1:7].mean(axis=(1, 3))
    vx = bx / (ix + 1e-3)

    gy = gy.reshape(h8, 8, w8, 8).transpose(0, 2, 1, 3)  # block-row, block-col, y-in-block, x
    by = gy[:, :, 7, :].mean(axis=2)
    iy = gy[:, :, 1:7, :].mean(axis=(2, 3))
    vy = by / (iy + 1e-3)

    ratio = 0.5 * (vx + vy)
    z_block = robust_z(np.log1p(ratio * 4.0))
    z = cv2.resize(z_block.astype(np.float32), (g.shape[1], g.shape[0]),
                   interpolation=cv2.INTER_NEAREST)
    return ratio.astype(np.float32), z


def spectral_period(hist: np.ndarray, max_q: int = 26) -> tuple[int, float, np.ndarray]:
    """Detect the period of a coefficient-magnitude histogram.

    Double quantisation with step ``q1`` leaves periodic gaps at multiples of
    ``q1`` in the histogram of the final coefficients. After detrending, the
    period shows up as a spectral peak at ``N / q1``.
    """
    h = np.asarray(hist, np.float32).ravel()
    if h.size < 32 or h.sum() < 200:
        return 0, 0.0, np.zeros(max_q + 1, np.float32)
    # drop DC bin, detrend slowly-varying content
    h = h[1:]
    h = h - cv2.blur(h.reshape(1, -1), (9, 1)).ravel()
    spec = np.abs(np.fft.rfft(h))
    scores = np.zeros(max_q + 1, np.float32)
    for q in range(2, max_q + 1):
        idx = int(round(h.size / q))
        if idx <= 2 or idx >= spec.size - 2:
            continue
        band = float(spec[idx]) + 0.5 * (float(spec[idx - 1]) + float(spec[idx + 1]))
        lo, hi = max(1, idx - 7), min(spec.size, idx + 8)
        local = float(np.median(spec[lo:hi])) + 1e-6
        scores[q] = band / local
    best_q = int(np.argmax(scores))
    return best_q, float(scores[best_q]), scores


def block_dct(gray: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """8x8 type-II DCT of every aligned block -> ``(h8, w8, 8, 8)``."""
    h8, w8 = gray.shape[0] // 8, gray.shape[1] // 8
    if h8 == 0 or w8 == 0:
        return np.zeros((0, 0, 8, 8), np.float32)
    g = np.clip(gray[: h8 * 8, : w8 * 8] * scale, 0, 255).astype(np.float32)
    blocks = g.reshape(h8, 8, w8, 8).transpose(0, 2, 1, 3)
    # level shift the DC term the way JPEG does
    blocks = blocks - 128.0
    return dctn(blocks, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)


#: AC coefficient positions (zig-zag order with zero-based 8x8 indices).
_ZIGZAG = [
    (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1),
    (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
]


def coeff_sample(dct: np.ndarray, mask: np.ndarray | None = None,
                 max_blocks: int = 4000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Flatten selected AC coefficients for histogram analysis."""
    if dct.size == 0:
        return np.zeros(0, np.float32)
    h8, w8 = dct.shape[:2]
    pts = np.argwhere(mask[: h8 * 8, : w8 * 8][::8, ::8]) if mask is not None else None
    if pts is not None and len(pts) > 0:
        rows = np.clip((pts[:, 0] // 8), 0, h8 - 1)
        cols = np.clip((pts[:, 1] // 8), 0, w8 - 1)
        idx = rows * w8 + cols
    else:
        idx = np.arange(h8 * w8)
    rng = rng or np.random.default_rng(0)
    if idx.size > max_blocks:
        idx = rng.choice(idx, size=max_blocks, replace=False)
    flat = dct.reshape(h8 * w8, 64)
    sel = np.stack([flat[idx, r * 8 + c] for r, c in _ZIGZAG], axis=1)
    return np.abs(sel).ravel()


def dq_scan(dct: np.ndarray, mask: np.ndarray | None = None,
            max_q: int = 26) -> tuple[int, float]:
    """Best double-quantisation period and its confidence for a region."""
    coefs = coeff_sample(dct, mask)
    coefs = coefs[(coefs > 0) & (coefs < 160)]
    if coefs.size < 900:
        return 0, 0.0
    hist = np.bincount(coefs.astype(np.int32), minlength=200)[:200].astype(np.float32)
    q, score, _ = spectral_period(hist, max_q=max_q)
    return q, score
