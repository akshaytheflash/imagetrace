"""Classical (model-free) evidence fusion.

A transparent weighted log-odds sum over the forensic maps. This is the
fallback when no learned checkpoint is present, and it is also blended with
the U-Net output so that an out-of-distribution image cannot be silently
hallucinated into a clean verdict.
"""

from __future__ import annotations

import numpy as np

from ..config import CHANNELS
from .stack import FeatureStack

#: Weights sum to 1.0. Kept beside the code so the report can print them.
CLASSICAL_WEIGHTS: dict[str, float] = {
    "noise_z": 0.22,
    "ela": 0.15,
    "seam": 0.14,
    "ghost_min": 0.12,
    "ghost_spread": 0.08,
    "cfa_resid": 0.08,
    "blockiness": 0.07,
    "resample": 0.06,
    "color_shift": 0.05,
    "hf_energy": 0.03,
    "edge_density": 0.00,
    "noise_var": 0.00,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def channel_contributions(stack: FeatureStack) -> dict[str, np.ndarray]:
    return {name: CLASSICAL_WEIGHTS.get(name, 0.0) * stack.maps[name] for name in CHANNELS}


def classical_score(stack: FeatureStack) -> np.ndarray:
    """Probability-like manipulation score in [0, 1].

    Calibrated against the image's own robust background distribution, not a
    fixed constant: the decision boundary sits at ``median + 2 robust sigma``.
    That keeps a clean, heavily compressed photo from flagging everywhere while
    still reacting to a genuinely foreign patch.
    """
    acc = np.zeros(stack.shape, np.float32)
    for name, w in CLASSICAL_WEIGHTS.items():
        if w:
            acc += w * stack.maps[name]
    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med))) * 1.4826
    # Floor the spread: a perfectly synthetic, uniform image has no local
    # statistics to speak of and should not produce a confident mask.
    scale = max(mad, 0.022)
    z = (acc - med) / scale
    return _sigmoid(1.55 * (z - 2.0)).astype(np.float32)
