"""Sensor-noise forensics.

A photo carries a roughly homogeneous noise floor (photon shot noise plus
read noise, modulated by ISO and demosaicing). Splices, synthesized fills and
strong denoising break that homogeneity. We estimate a local noise standard
deviation with a median-based high-pass that keeps edges intact, then express
it as a robust z-score over the image.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io_utils import box_mean, robust_z


def noise_residual(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving high-pass: pixel minus 3x3 median."""
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(g8, 3).astype(np.float32) / 255.0
    return gray - med


def noise_maps(gray: np.ndarray, win: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sigma, z)``: local noise std and its robust z-score."""
    res = noise_residual(gray)
    # Ignore the immediate neighbourhood of strong edges: their residual is
    # dominated by quantisation of the median filter, not by noise.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    w = 1.0 / (1.0 + (edge * 40.0) ** 2)
    m = box_mean(res, win)
    m2 = box_mean(res * res, win)
    var = np.maximum(m2 - m * m, 0.0)
    conf = box_mean(w, win)
    sigma = np.sqrt(var) / np.maximum(conf, 0.25)
    # Residual after resize compression is tiny; express in 8-bit units for
    # readability in the report.
    sigma = sigma * 255.0
    return sigma, robust_z(sigma)


def noise_floor(sigma: np.ndarray, mask: np.ndarray) -> float:
    vals = sigma[mask]
    if vals.size == 0:
        return float(np.median(sigma))
    return float(np.median(vals))
