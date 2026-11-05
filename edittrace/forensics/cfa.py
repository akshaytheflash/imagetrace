"""CFA / demosaicing residual.

Almost every camera sensor samples one colour per photosite and interpolates
the rest. That leaves a 2x2-periodic correlation between channels. Pasted
content from a rendered image, a screengrab, or a heavily processed asset does
not share the host's CFA fingerprint.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io_utils import box_mean, robust_norm, robust_z


def cfa_map(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(magnitude, z)`` maps for the 2x2 CFA residual."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = (g - 0.5 * (r + b)) * 255.0

    n = min(d.shape[0] // 2, d.shape[1] // 2)
    s00 = d[0: 2 * n: 2, 0: 2 * n: 2]
    s01 = d[0: 2 * n: 2, 1: 2 * n: 2]
    s10 = d[1: 2 * n: 2, 0: 2 * n: 2]
    s11 = d[1: 2 * n: 2, 1: 2 * n: 2]
    checker = 0.25 * (s00 + s11 - s01 - s10)      # half resolution

    mag = box_mean(np.abs(checker), 5)
    mag = cv2.resize(mag, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Mild denoise so texture-induced chroma edges do not dominate.
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    return mag, robust_z(robust_norm(mag))
