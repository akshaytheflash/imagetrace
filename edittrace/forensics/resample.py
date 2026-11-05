"""Resampling periodicity.

Scaling or rotating a patch interpolates on a regular lattice. The second
derivative of the interpolated signal then carries periodic variance at the
resampling period. We measure the local normalised autocorrelation of the
second derivative over small lags; a strong off-zero peak is the signature.
"""

from __future__ import annotations

import numpy as np

from ..io_utils import box_mean


def _second_derivative(x: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        d = x[:, 2:] + x[:, :-2] - 2.0 * x[:, 1:-1]
        return np.pad(d, ((0, 0), (1, 1)), mode="edge")
    d = x[2:, :] + x[:-2, :] - 2.0 * x[1:-1, :]
    return np.pad(d, ((1, 1), (0, 0)), mode="edge")


def resample_map(gray: np.ndarray, win: int = 11,
                 lags: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(periodicity, z)`` in [0, 1]."""
    best = np.zeros_like(gray, np.float32)
    energy_total = np.zeros_like(gray, np.float32)

    for axis in (0, 1):
        d2 = _second_derivative(gray, axis)
        power = box_mean(d2 * d2, win) + 1e-9
        energy_total += power
        for lag in lags:
            shifted = np.roll(d2, lag, axis=axis)
            corr = box_mean(d2 * shifted, win) / power
            best = np.maximum(best, np.abs(corr))

    # Weight by local second-derivative energy: flat areas produce spurious
    # correlations that carry no resampling information.
    reliability = box_mean(energy_total, win)
    rel_norm = reliability / (np.percentile(reliability, 95) + 1e-9)
    score = np.clip(best * np.clip(rel_norm, 0, 1), 0, 1)
    return score.astype(np.float32), score.astype(np.float32)
