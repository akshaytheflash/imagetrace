"""Configuration objects and tuned constants.

Every numeric knob that affects a forensic conclusion lives here so that the
weights used in a report can be traced back to a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Forensic channels produced by :mod:`edittrace.forensics.stack`, in order.
CHANNELS: tuple[str, ...] = (
    "ela",            # error level analysis, recompression residual
    "noise_z",        # sensor-noise inconsistency, robust z-score
    "noise_var",      # absolute local noise standard deviation
    "ghost_min",      # min over quality sweep of JPEG ghost residual
    "ghost_spread",   # spread of ghost residual across qualities
    "blockiness",     # 8x8 block-boundary energy, block-grid mismatch
    "cfa_resid",      # Bayer/CFA interpolation residual
    "resample",       # resampling periodicity (2nd-derivative autocorrelation)
    "seam",           # texture edges coinciding with statistical discontinuity
    "hf_energy",      # high-frequency energy
    "color_shift",    # local chromatic anomaly (LAB ab)
    "edge_density",   # gradient density
)

#: Luma and two context channels are appended for the learned fusion model.
MODEL_CHANNELS: tuple[str, ...] = CHANNELS + (
    "luma", "luma_local_mean", "luma_local_std",
    "noise_contrast", "hf_contrast", "ela_contrast",
)


@dataclass
class Config:
    """Runtime configuration for the analysis pipeline."""

    # --- signal extraction -------------------------------------------------
    ela_quality: int | None = None          # None -> use the JPEG table estimate
    ghost_qualities: tuple[int, ...] = (55, 60, 65, 70, 75, 80, 85, 90, 95)
    max_side: int = 1600                    # downscale absurdly large inputs

    # --- region hypotheses ------------------------------------------------
    min_region_frac: float = 0.0015         # of total pixels
    max_regions: int = 8
    mask_threshold: float = 0.58

    # --- fusion -----------------------------------------------------------
    model_path: str | None = None           # None -> package default / classical
    precedence_path: str | None = None      # None -> package default / rule-based
    model_weight: float = 0.80
    device: str = "cpu"
    tile: int = 640                         # inference tile for large images

    # --- ordering ---------------------------------------------------------
    # Weights of the four precedence cues. They sum to 1 and are printed in
    # every report so a reader can audit the ranking.
    order_weights: dict[str, float] = field(
        default_factory=lambda: {
            "compression_lineage": 0.30,
            "recompression_smoothing": 0.24,
            "seam_truncation": 0.18,
            "seam_blend": 0.14,
            "noise_floor": 0.14,
        }
    )
    precedence_eps: float = 0.06            # |score| below this is "undecided"

    # --- provenance -------------------------------------------------------
    provenance_penalty: float = 0.75        # max score removed by audit

    def resolved_model_path(self) -> Path | None:
        if self.model_path is not None:
            p = Path(self.model_path)
            return p if p.exists() else None
        default = Path(__file__).parent / "weights" / "edittrace_unet.pt"
        return default if default.exists() else None

    def resolved_precedence_path(self) -> Path | None:
        if self.precedence_path is not None:
            p = Path(self.precedence_path)
            return p if p.exists() else None
        default = Path(__file__).parent / "weights" / "precedence.npz"
        return default if default.exists() else None
