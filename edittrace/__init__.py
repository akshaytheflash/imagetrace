"""EditTrace: localize edits, infer their order, and audit provenance.

Public entry point is :func:`edittrace.pipeline.analyze`, which returns a
:class:`edittrace.pipeline.Trace` and can render a self-contained HTML report.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import analyze, Trace  # noqa: E402,F401

__all__ = ["analyze", "Trace", "__version__"]
