"""Assemble the dense forensic feature stack."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CHANNELS, MODEL_CHANNELS, Config
from ..io_utils import ImageBundle, box_mean, box_std, robust_norm, robust_z
from .cfa import cfa_map
from .dct import blockiness_map, ghost_maps
from .ela import ela_map
from .noise import noise_maps
from .resample import resample_map
from .seam import seam_map


@dataclass
class FeatureStack:
    bundle: ImageBundle
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    ghost_best_q: np.ndarray | None = None
    ela_quality: int = 90
    host_quality: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.bundle.gray.shape[:2]

    def tensor(self) -> np.ndarray:
        """``(18, H, W)`` float32 stack for the fusion U-Net.

        Twelve forensic channels, three context channels (luma, local mean,
        local texture), and three explicit contrast channels that subtract a
        wide local baseline. The contrast pair is what lets the network ask
        "different from its surroundings?" rather than "high or low?".
        """
        chans = [self.maps[name] for name in CHANNELS]
        chans.append(self.bundle.gray)
        chans.append(self.raw["luma_mean"])
        chans.append(self.raw["luma_std"])
        chans.append(self.raw["noise_contrast"])
        chans.append(self.raw["hf_contrast"])
        chans.append(self.raw["ela_contrast"])
        return np.stack(chans, axis=0).astype(np.float32)

    def display(self, name: str) -> np.ndarray:
        """8-bit visualisation for reports (signed maps become magnitude)."""
        x = self.maps[name]
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def build_features(bundle: ImageBundle, cfg: Config | None = None) -> FeatureStack:
    cfg = cfg or Config()
    gray = bundle.gray
    rgb = bundle.rgb

    host_q, q_err = bundle.jpeg_quality
    ela_q = cfg.ela_quality or (host_q if host_q and q_err < 0.25 else 90)
    ela_raw, ela = ela_map(bundle.bgr, quality=ela_q)

    sigma, nz = noise_maps(gray)
    gmin, gspread, best_q = ghost_maps(bundle.bgr, qualities=cfg.ghost_qualities)
    _, block_z = blockiness_map(gray)
    cfa_mag, cfa_z = cfa_map(rgb)
    resample, _ = resample_map(gray)
    seam, disc = seam_map(gray, ela, nz)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    hf = box_mean(np.abs(lap), 5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = box_mean(cv2.magnitude(gx, gy), 5)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    chroma_dev = np.sqrt(
        (lab[..., 1] - box_mean(lab[..., 1], 31)) ** 2
        + (lab[..., 2] - box_mean(lab[..., 2], 31)) ** 2
    )
    color = box_mean(chroma_dev, 5)

    raw = {
        "ela": ela_raw,
        "noise_z": nz,
        "noise_var": sigma,
        "ghost_min": gmin,
        "ghost_spread": gspread,
        "blockiness": block_z,
        "cfa_resid": cfa_mag,
        "resample": resample,
        "seam": seam,
        "hf_energy": hf,
        "color_shift": color,
        "edge_density": edges,
        "discontinuity": disc,
    }

    # A pasted region is not "high ghost residual", it is residually *different*
    # from its neighbourhood. Compare against a local (wide) baseline instead of
    # normalising the absolute value.
    ghost_dev = np.abs(gmin - box_mean(gmin, 41))
    maps = {
        "ela": robust_norm(ela),
        "noise_z": robust_norm(np.abs(nz)),
        "noise_var": robust_norm(sigma),
        "ghost_min": robust_norm(ghost_dev),
        "ghost_spread": robust_norm(gspread),
        "blockiness": robust_norm(np.abs(block_z)),
        "cfa_resid": robust_norm(cfa_mag),
        "resample": robust_norm(resample),
        "seam": seam,
        "hf_energy": robust_norm(hf),
        "color_shift": robust_norm(color),
        "edge_density": robust_norm(edges),
    }
    # raw ghost baseline kept for region cards
    raw["ghost_baseline"] = box_mean(gmin, 41)
    raw["noise_abs_z"] = np.abs(nz)
    raw["luma_mean"] = box_mean(gray, 31)
    raw["luma_std"] = np.clip(box_std(gray, 15) * 6.0, 0, 1)
    raw["noise_contrast"] = robust_norm(np.abs(sigma - box_mean(sigma, 65)))
    raw["hf_contrast"] = robust_norm(np.abs(hf - box_mean(hf, 65)))
    raw["ela_contrast"] = robust_norm(np.abs(ela - box_mean(ela, 65)))

    return FeatureStack(
        bundle=bundle,
        raw=raw,
        maps={k: v.astype(np.float32) for k, v in maps.items()},
        ghost_best_q=best_q,
        ela_quality=ela_q,
        host_quality=host_q,
    )
