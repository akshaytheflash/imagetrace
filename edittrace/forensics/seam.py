"""Manipulation-boundary evidence.

A splice boundary is where two independent statistics meet. Texture edges
alone are everywhere, and statistical anomalies alone are blobby; their
product is a much sharper localiser. We also keep the raw discontinuity field
so region cards can quote it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io_utils import box_mean, robust_norm


def seam_map(gray: np.ndarray, ela: np.ndarray, noise_z: np.ndarray,
             win: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(seam, discontinuity)`` maps normalised to [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge = robust_norm(box_mean(edge, 3), 5, 99)

    # Discontinuity: high-frequency part of ELA plus noise inconsistency.
    ela_local = np.abs(ela - box_mean(ela, win))
    disc = 0.6 * robust_norm(ela_local) + 0.4 * robust_norm(np.abs(noise_z))
    seam = edge * box_mean(disc, 5)
    seam = box_mean(seam, 3)
    return robust_norm(seam).astype(np.float32), robust_norm(disc).astype(np.float32)
