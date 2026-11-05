"""Region taxonomy: clone, splice, resampled insert, inpaint/removal."""

from __future__ import annotations

import cv2
import numpy as np

from .extract import Region


def find_clone_source(bgr: np.ndarray, region: Region,
                      max_side: int = 420) -> tuple[float, tuple[int, int] | None]:
    """Template-match a region against the rest of the image.

    Returns ``(ncc, source_xy)``. Rotation and scale are not searched here;
    the detector asks a narrower question: was this exact texture copied?
    """
    h, w = bgr.shape[:2]
    x, y, bw, bh = region.bbox
    if bw < 12 or bh < 12 or bw > 0.6 * w or bh > 0.6 * h:
        return 0.0, None

    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # gradient magnitude: invariant to flat colour shifts, sensitive to texture
    feat = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    feat = cv2.GaussianBlur(feat, (0, 0), 1.0)

    sx, sy = int(round(x * s)), int(round(y * s))
    tx, ty = int(round(bw * s)), int(round(bh * s))
    tx, ty = max(8, min(tx, small.shape[1] - 1)), max(8, min(ty, small.shape[0] - 1))
    if sx + tx > small.shape[1] or sy + ty > small.shape[0]:
        return 0.0, None
    templ = feat[sy:sy + ty, sx:sx + tx].copy()
    if templ.size == 0 or templ.std() < 1e-3:
        return 0.0, None

    m = region.mask
    tmask = cv2.resize(m, (tx, ty), interpolation=cv2.INTER_NEAREST)
    tmask = (tmask > 127).astype(np.uint8) * 255
    if tmask.sum() == 0:
        return 0.0, None

    try:
        res = cv2.matchTemplate(feat, templ, cv2.TM_CCOEFF_NORMED, mask=tmask)
    except cv2.error:
        res = cv2.matchTemplate(feat, templ, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=0.0)
    # suppress the region itself and its immediate halo
    cx, cy = sx + tx // 2, sy + ty // 2
    yy, xx = np.mgrid[0:res.shape[0], 0:res.shape[1]]
    dist = np.sqrt((xx + tx / 2 - cx) ** 2 + (yy + ty / 2 - cy) ** 2)
    res[dist < 0.75 * max(tx, ty)] = -1.0
    _, best, _, best_loc = cv2.minMaxLoc(res)
    if best_loc is None or best <= -0.5:
        return 0.0, None
    src = (int(best_loc[0] / s), int(best_loc[1] / s))
    return float(np.clip(best, 0, 1)), src


def classify_region(region: Region, bgr: np.ndarray) -> Region:
    """Score the region against four edit archetypes and record the winner."""
    st = region.stats
    clone_ncc, clone_src = find_clone_source(bgr, region)
    region.stats["clone_ncc"] = clone_ncc
    if clone_ncc >= 0.9:
        region.copy_source = clone_src

    noise_eff = st["noise_eff"]
    hf_eff = st["hf_eff"]
    resample = st["resample"]
    ghost_mismatch = abs(st["ghost_q"] - st["ghost_q_ring"]) >= 10 and st["ghost_q"] > 0

    scores = {
        "copy_move": float(np.clip((clone_ncc - 0.86) / 0.12, 0, 1)),
        "inpaint": float(0.6 * np.clip(-noise_eff / 2.5, 0, 1)
                         + 0.4 * np.clip(-hf_eff / 2.5, 0, 1)),
        "resampled_insert": float(np.clip((resample - 0.18) / 0.30, 0, 1)),
        "splice": float(0.55 * np.clip(abs(noise_eff) / 2.5, 0, 1)
                        + 0.25 * np.clip(abs(st["ela_eff"]) / 2.0, 0, 1)
                        + 0.20 * (1.0 if ghost_mismatch else 0.0)),
    }
    # Every hypothesis keeps a floor so a weak classification says so.
    scores = {k: v + 0.02 for k, v in scores.items()}
    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}
    kind = max(probs, key=probs.get)

    # A cloned texture is a hard, high-precision signal; let it dominate.
    if clone_ncc >= 0.94:
        kind = "copy_move"
        probs["copy_move"] = max(probs["copy_move"], 0.7)

    region.kind = kind
    region.kind_scores = {k: float(v) for k, v in probs.items()}
    region.stats["kind_conf"] = float(probs[kind])
    return region
