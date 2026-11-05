"""Turn a probability map into a small set of auditable region hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Config
from ..forensics.dct import block_dct, dq_scan
from ..forensics.stack import FeatureStack
from ..io_utils import box_mean


@dataclass
class Region:
    id: int
    mask: np.ndarray                       # uint8 0/255
    bbox: tuple[int, int, int, int]
    area_px: int
    area_frac: float
    mean_prob: float
    confidence: float
    stats: dict[str, float] = field(default_factory=dict)
    kind: str = "unknown"
    kind_scores: dict[str, float] = field(default_factory=dict)
    copy_source: tuple[int, int] | None = None
    evidence: list[dict] = field(default_factory=list)
    attribution: dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


def _effect(in_vals: np.ndarray, ring_vals: np.ndarray) -> tuple[float, float, float]:
    if in_vals.size == 0 or ring_vals.size == 0:
        return 0.0, 0.0, 0.0
    mi, mr = float(in_vals.mean()), float(ring_vals.mean())
    sd = float(ring_vals.std()) + 1e-6
    return mi, mr, float(np.clip((mi - mr) / sd, -8, 8))


def region_mask_from_prob(prob: np.ndarray, cfg: Config) -> np.ndarray:
    """Hysteresis threshold + cleanup, with a guard against over-segmentation."""
    t = cfg.mask_threshold
    high = (prob >= t).astype(np.uint8) * 255
    low = (prob >= max(0.30, t - 0.16)).astype(np.uint8) * 255
    if high.sum() == 0:
        return np.zeros_like(high)

    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Keep high-confidence components, then let them grow through the
    # lower-confidence halo. This recovers soft feathered boundaries without
    # spawning regions in noise.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(high, 8)
    min_area = max(48, int(cfg.min_region_frac * prob.size))
    keep = np.zeros_like(high)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return keep

    grown = keep.copy()
    for _ in range(6):
        grown = cv2.dilate(grown, np.ones((5, 5), np.uint8))
        grown = cv2.bitwise_and(grown, low)
        grown = cv2.bitwise_or(grown, keep)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Fill interior holes: a large uniform pasted area produces a ring of
    # evidence at its boundary but little in the middle.
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grown, cnts, -1, 255, -1)
    return grown


def _perimeter(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    k = np.ones((thickness, thickness), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)


def describe_region(stack: FeatureStack, mask: np.ndarray, rid: int,
                    dct_cache: np.ndarray | None = None,
                    global_dq: tuple[int, float] = (0, 0.0),
                    mean_prob: float = 1.0,
                    cfg: Config | None = None) -> Region:
    """Compute the full descriptor set for one mask.

    Shared by the detector and by the ordering evaluation (which feeds
    ground-truth masks here to isolate ordering from localization).
    """
    cfg = cfg or Config()
    mask = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("empty mask")
    x, y = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
    area = int((mask > 0).sum())
    if dct_cache is None:
        dct_cache = block_dct(stack.bundle.gray)

    in_m = mask > 0
    ring = (cv2.dilate(mask, np.ones((19, 19), np.uint8)) > 0) & \
           ~(cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0)
    if ring.sum() < 32:
        ring = (cv2.dilate(mask, np.ones((35, 35), np.uint8)) > 0) & ~in_m

    raw = stack.raw
    sigma_in, sigma_ring, noise_eff = _effect(raw["noise_var"][in_m], raw["noise_var"][ring])
    ela_in, ela_ring, ela_eff = _effect(raw["ela"][in_m], raw["ela"][ring])
    hf_in, hf_ring, hf_eff = _effect(raw["hf_energy"][in_m], raw["hf_energy"][ring])
    col_in, col_ring, col_eff = _effect(raw["color_shift"][in_m], raw["color_shift"][ring])
    cfa_in, cfa_ring, cfa_eff = _effect(raw["cfa_resid"][in_m], raw["cfa_resid"][ring])
    blk_in, blk_ring, blk_eff = _effect(raw["blockiness"][in_m], raw["blockiness"][ring])
    gmin_in, gmin_ring, ghost_eff = _effect(raw["ghost_min"][in_m], raw["ghost_min"][ring])
    gspread_in = float(raw["ghost_spread"][in_m].mean())

    best_q_vals = stack.ghost_best_q[in_m] if stack.ghost_best_q is not None else np.array([0])
    ghost_q_mode = int(np.bincount(best_q_vals.astype(np.int32)).argmax())
    if stack.ghost_best_q is not None:
        ring_q = stack.ghost_best_q[ring]
        ghost_q_ring = int(np.bincount(ring_q.astype(np.int32)).argmax())
    else:
        ghost_q_ring = 0

    perim = _perimeter(mask, 3) > 0
    seam_vals = raw["seam"][perim]
    seam_mean = float(seam_vals.mean()) if seam_vals.size else 0.0
    seam_hot = float((seam_vals > 0.55).mean()) if seam_vals.size else 0.0
    seam_ring_mean = float(raw["seam"][ring].mean())

    dq_q, dq_s = dq_scan(dct_cache, mask) if area > 1600 else (0, 0.0)

    stats = {
        "noise_in": sigma_in, "noise_ring": sigma_ring, "noise_eff": noise_eff,
        "ela_in": ela_in, "ela_ring": ela_ring, "ela_eff": ela_eff,
        "hf_in": hf_in, "hf_ring": hf_ring, "hf_eff": hf_eff,
        "color_eff": col_eff, "cfa_eff": cfa_eff, "cfa_in": cfa_in,
        "block_eff": blk_eff, "ghost_eff": ghost_eff,
        "ghost_spread": gspread_in, "ghost_q": float(ghost_q_mode),
        "ghost_q_ring": float(ghost_q_ring),
        "seam_mean": seam_mean, "seam_ring": seam_ring_mean, "seam_hot": seam_hot,
        "resample": float(raw["resample"][in_m].mean()),
        "dq_q": float(dq_q), "dq_score": float(dq_s),
        "dq_global_q": float(global_dq[0]), "dq_global_score": float(global_dq[1]),
        "host_q": float(stack.host_quality) if stack.host_quality else 0.0,
        "area_frac": float(area) / mask.size,
        "mean_prob": float(mean_prob),
    }

    conf = float(np.clip(
        0.55 * mean_prob
        + 0.20 * min(1.0, abs(noise_eff) / 2.5)
        + 0.15 * min(1.0, abs(ela_eff) / 2.0)
        + 0.10 * min(1.0, (area / mask.size) / 0.06),
        0.03, 0.985))

    return Region(
        id=rid, mask=mask, bbox=(x, y, bw, bh), area_px=area,
        area_frac=float(area) / mask.size, mean_prob=float(mean_prob),
        confidence=conf, stats=stats,
    )


def extract_regions(stack: FeatureStack, prob: np.ndarray,
                    cfg: Config | None = None,
                    dct_cache: np.ndarray | None = None) -> tuple[list[Region], np.ndarray]:
    cfg = cfg or Config()
    h, w = prob.shape
    binary = region_mask_from_prob(prob, cfg)
    if binary.sum() == 0:
        return [], binary

    if dct_cache is None:
        dct_cache = block_dct(stack.bundle.gray)
    global_dq = dq_scan(dct_cache)

    n, labels, comp_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(48, int(cfg.min_region_frac * prob.size))
    candidates = []
    for i in range(1, n):
        area = int(comp_stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mean_p = float(prob[labels == i].mean())
        candidates.append((i, area, mean_p))
    candidates.sort(key=lambda c: (c[2] * np.sqrt(c[1])), reverse=True)
    candidates = candidates[: cfg.max_regions]

    regions: list[Region] = []
    for rid, (i, area, mean_p) in enumerate(candidates, start=1):
        m = ((labels == i).astype(np.uint8)) * 255
        regions.append(describe_region(stack, m, rid, dct_cache, global_dq,
                                       mean_prob=mean_p, cfg=cfg))
    return regions, binary
