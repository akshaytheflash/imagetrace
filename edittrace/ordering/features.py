"""Feature construction for the learned precedence model.

The vector is **antisymmetric**: swapping A and B negates every component, so
``p(A before B) = 1 - p(B before A)`` by construction. That removes a whole
class of ranking inconsistencies before the graph is even built.
"""

from __future__ import annotations

import numpy as np

from ..forensics.stack import FeatureStack
from ..regions.extract import Region
from .cues import PairEvidence

KINDS = ("splice", "copy_move", "inpaint", "resampled_insert", "unknown")

FEATURE_NAMES = (
    ["compression_lineage", "recompression_smoothing", "seam_truncation",
     "seam_blend", "noise_floor"]
    + ["log_noise_ratio", "log_hf_ratio", "log_ela_ratio", "resample_ratio",
       "dq_diff", "seam_diff", "log_area_ratio", "conf_diff"]
    + [f"kind_{k}" for k in KINDS]
)


def _safe_log_ratio(a: float, b: float) -> float:
    return float(np.log((max(a, 0.0) + 1e-3) / (max(b, 0.0) + 1e-3)))


def pair_vector(ev: PairEvidence, a: Region, b: Region, stack: FeatureStack) -> np.ndarray:
    sa, sb = a.stats, b.stats
    cues = [ev.cues.get(k, 0.0) for k in
            ("compression_lineage", "recompression_smoothing",
             "seam_truncation", "seam_blend", "noise_floor")]
    ratios = [
        _safe_log_ratio(sa["noise_in"], sb["noise_in"]),
        _safe_log_ratio(sa["hf_in"], sb["hf_in"]),
        _safe_log_ratio(sa["ela_in"], sb["ela_in"]),
        float(np.clip(sa["resample"] - sb["resample"], -1, 1)),
        float(np.clip((sa["dq_score"] - sb["dq_score"]) / 3.0, -1, 1)),
        float(np.clip((sa["seam_mean"] - sb["seam_mean"]) / 0.5, -1, 1)),
        float(np.clip(_safe_log_ratio(sa["area_frac"], sb["area_frac"]), -2, 2)),
        float(np.clip(a.confidence - b.confidence, -1, 1)),
    ]
    kinds = [1.0 if a.kind == k else 0.0 for k in KINDS]
    kinds = [x - (1.0 if b.kind == k else 0.0) for x, k in zip(kinds, KINDS)]
    return np.asarray(cues + ratios + kinds, dtype=np.float32)


class PrecedenceModel:
    def __init__(self, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        self.coef = coef.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.scale = np.maximum(scale.astype(np.float32), 1e-6)

    def margin(self, f: np.ndarray) -> float:
        z = (f - self.mean) / self.scale
        return float(np.dot(self.coef, z))

    def prob(self, f: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.margin(f), -30, 30))))

    def save(self, path) -> None:
        np.savez(path, coef=self.coef, mean=self.mean, scale=self.scale)


def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 900,
                 lr: float = 0.35, l2: float = 2e-3) -> PrecedenceModel:
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    Z = (X - mean) / scale
    w = np.zeros(Z.shape[1], np.float32)
    n = len(Z)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return PrecedenceModel(w, mean, scale)


def load(path) -> PrecedenceModel | None:
    try:
        d = np.load(path)
        return PrecedenceModel(d["coef"], d["mean"], d["scale"])
    except Exception:
        return None
