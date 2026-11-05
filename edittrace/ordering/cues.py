"""Pairwise precedence cues.

``s > 0`` means "A was applied before B". Every cue returns a signed value in
[-1, 1] and records whether it was *available* for that pair; unavailable cues
are dropped and the weights renormalised. This matters: a PNG with no JPEG
history should not silently inherit a compression-based ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Config
from ..forensics.stack import FeatureStack
from ..io_utils import box_mean
from ..regions.extract import Region

EPS = 1e-6


def _perim(mask: np.ndarray, thickness: int = 5) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    k = np.ones((thickness, thickness), np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k) > 0


def _adjacent_band(a: np.ndarray, b: np.ndarray, depth: int = 3) -> np.ndarray:
    """Pixels just inside ``a`` that sit within ``depth`` px of ``b``."""
    near_b = cv2.dilate(b.astype(np.uint8) * 255, np.ones((2 * depth + 1,) * 2, np.uint8)) > 0
    return near_b & (a > 0)


@dataclass
class PairEvidence:
    a: int
    b: int
    score: float = 0.0
    cues: dict[str, float] = field(default_factory=dict)
    available: dict[str, bool] = field(default_factory=dict)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def pair_evidence(a: Region, b: Region, stack: FeatureStack,
                  cfg: Config, global_dq: tuple[int, float]) -> PairEvidence:
    ev = PairEvidence(a=a.id, b=b.id)
    sa, sb = a.stats, b.stats
    h, w = stack.shape

    # ------------------------------------------------------------------
    # 1. Compression lineage.
    # If a whole image was saved at q1 before an object was inserted, every
    # pixel that existed at q1 carries a double-quantisation signature with
    # period q1. The inserted object does not. Stronger DQ therefore means
    # "was here earlier". Gated on the host actually showing the effect.
    # ------------------------------------------------------------------
    ga_q, ga_s = sa["dq_q"], sa["dq_score"]
    gb_q, gb_s = sb["dq_q"], sb["dq_score"]
    dq_informative = global_dq[1] >= 1.7 or max(ga_s, gb_s) >= 2.2
    ev.available["compression_lineage"] = bool(dq_informative)
    if dq_informative:
        delta = ga_s - gb_s
        s = _clip(delta / 3.2, -0.85, 0.85)
        # Same period as the host's earlier save adds a little confidence.
        if ga_q > 0 and ga_q == global_dq[0] and ga_s > 2.2:
            s = _clip(s + 0.12, -0.85, 0.85)
        if gb_q > 0 and gb_q == global_dq[0] and gb_s > 2.2:
            s = _clip(s - 0.12, -0.85, 0.85)
        if ga_s < 2.0 and gb_s < 2.0:
            s = 0.0
        ev.cues["compression_lineage"] = s

    # ------------------------------------------------------------------
    # 2. Recompression smoothing.
    # Each extra save attenuates high-frequency detail. If A has passed
    # through more saves than B, A's noise floor, high-frequency energy and
    # ELA residual are all lower. Ratios are taken against the local ring so
    # content does not dominate.
    # ------------------------------------------------------------------
    ev.available["recompression_smoothing"] = (
        stack.host_quality is not None
        and a.kind in {"splice", "resampled_insert"}
        and b.kind in {"splice", "resampled_insert"}
    )
    if ev.available["recompression_smoothing"]:
        def rel(key):
            a_v, b_v = sa[key], sb[key]
            denom = abs(a_v) + abs(b_v) + 1e-6
            return (a_v - b_v) / denom

        smooth = (0.45 * rel("noise_in") + 0.35 * rel("hf_in") + 0.20 * rel("ela_in"))
        # A smoother than B -> A was re-saved more often -> A earlier.
        ev.cues["recompression_smoothing"] = _clip(-1.5 * smooth, -0.8, 0.8)

    # ------------------------------------------------------------------
    # 3. Seam truncation.
    # An edit applied on top erases the seam of what is underneath along the
    # shared boundary. Compare each region's seam energy on the shared arc to
    # its seam energy everywhere else.
    # ------------------------------------------------------------------
    ma, mb = a.mask, b.mask
    adj_a = _adjacent_band(ma, mb, 4)
    adj_b = _adjacent_band(mb, ma, 4)
    ev.available["seam_truncation"] = bool(adj_a.sum() > 40 and adj_b.sum() > 40)
    if ev.available["seam_truncation"]:
        seam = stack.raw["seam"]
        pa, pb = _perim(ma > 0), _perim(mb > 0)
        base_a = float(seam[pa].mean()) + EPS
        base_b = float(seam[pb].mean()) + EPS
        arc_a = float(seam[adj_a].mean())
        arc_b = float(seam[adj_b].mean())
        trunc_a = (base_a - arc_a) / base_a
        trunc_b = (base_b - arc_b) / base_b
        ev.cues["seam_truncation"] = _clip(1.6 * (trunc_a - trunc_b), -0.9, 0.9)

    # ------------------------------------------------------------------
    # 3. Seam blending.
    # A feathered paste bleeds the top layer's noise into the pixels it
    # covers. Whichever region better explains the shared band's noise is the
    # one that was applied last.
    # ------------------------------------------------------------------
    ev.available["seam_blend"] = ev.available["seam_truncation"]
    if ev.available["seam_blend"]:
        sigma = stack.raw["noise_var"]
        n_band = float(sigma[adj_a].mean())
        n_a = float(sigma[ma > 0].mean()) + EPS
        n_b = float(sigma[mb > 0].mean()) + EPS
        da, db = abs(n_band - n_a), abs(n_band - n_b)
        ev.cues["seam_blend"] = _clip(0.75 * (da - db) / (da + db + EPS), -0.75, 0.75)

    # ------------------------------------------------------------------
    # 4. Noise floor prior (weak).
    # Foreign inserts usually raise the local noise floor; inpainted fills
    # lower it. Only a tie-breaker, hence the small weight.
    # ------------------------------------------------------------------
    ev.available["noise_floor"] = True
    d = sa["noise_in"] - sb["noise_in"]
    ev.cues["noise_floor"] = _clip(-d / 5.0, -0.45, 0.45)

    weights = cfg.order_weights
    num = sum(weights[k] * v for k, v in ev.cues.items())
    den = sum(weights[k] for k, ok in ev.available.items() if ok and k in ev.cues)
    ev.score = _clip(num / den) if den > 0 else 0.0
    return ev


def all_pair_evidence(regions: list[Region], stack: FeatureStack,
                      cfg: Config) -> list[PairEvidence]:
    if len(regions) < 2:
        return []
    dct_q = int(regions[0].stats.get("dq_global_q", 0))
    dct_s = float(regions[0].stats.get("dq_global_score", 0.0))
    out = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            out.append(pair_evidence(regions[i], regions[j], stack, cfg, (dct_q, dct_s)))
    return out
