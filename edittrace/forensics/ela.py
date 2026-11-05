"""Error Level Analysis.

Re-encode at the file's own quality (or 90 for non-JPEG sources) and look at
the per-pixel reconstruction error. Regions with a different compression
history converge at a different rate, so the residual is locally elevated or
suppressed.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io_utils import box_mean


def ela_map(bgr: np.ndarray, quality: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ela_raw, ela_scaled)``.

    ``ela_raw`` is the max-channel absolute error in 8-bit units (0..255).
    ``ela_scaled`` is a display/normalised map in [0, 1].
    """
    q = int(np.clip(quality, 30, 100))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        zeros = np.zeros(bgr.shape[:2], np.float32)
        return zeros, zeros
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
    raw = np.abs(bgr.astype(np.float32) - recompressed).max(axis=2)

    # Local mean suppresses single-pixel ringing; normalise by a high
    # percentile so the map is comparable across images.
    smooth = box_mean(raw, 3)
    scale = float(np.percentile(smooth, 99.0))
    scaled = smooth / max(scale, 1.5)
    return raw, np.clip(scaled, 0.0, 1.5).astype(np.float32)
