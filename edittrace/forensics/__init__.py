"""Forensic signal extractors.

Each module is independent and returns dense float32 maps at image
resolution. Nothing here depends on the neural model, which keeps the
classical pipeline usable (and auditable) on its own.
"""

from .stack import FeatureStack, build_features  # noqa: F401
from .score import classical_score, channel_contributions  # noqa: F401
