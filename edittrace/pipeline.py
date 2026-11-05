"""End-to-end pipeline: image in, trace out."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CHANNELS, Config
from .explain.evidence import attribute_channels, build_evidence
from .forensics.dct import block_dct, dq_scan
from .forensics.score import CLASSICAL_WEIGHTS, classical_score
from .forensics.stack import FeatureStack, build_features
from .io_utils import ImageBundle, bundle_from_array, load_image
from .models.fusion import blend_scores, load_if_available
from .ordering.cues import PairEvidence, all_pair_evidence
from .ordering.topo import OrderResult, order_from_evidence
from .ordering.train_order import score_pairs
from .provenance.audit import ProvenanceAudit, audit
from .regions.classify import classify_region
from .regions.extract import Region, extract_regions


@dataclass
class Trace:
    bundle: ImageBundle
    features: FeatureStack
    prob: np.ndarray
    classical: np.ndarray
    model_prob: np.ndarray | None
    regions: list[Region]
    order: OrderResult
    pairs: list[PairEvidence]
    provenance: ProvenanceAudit
    mask: np.ndarray
    meta: dict = field(default_factory=dict)

    def render_html(self, path: str | Path) -> Path:
        from .explain.report import render_report
        return render_report(self, path)

    def to_dict(self) -> dict:
        def num(x, nd=4):
            return round(float(x), nd)

        return {
            "image": {
                "name": self.bundle.name,
                "width": self.bundle.w,
                "height": self.bundle.h,
                "bytes": self.bundle.filesize,
                "sha256": self.meta.get("sha256"),
                "format": self.bundle.tags.get("_format"),
            },
            "analysis": {
                "manipulated": self.meta.get("manipulated"),
                "edited_area_frac": num(self.mask.mean()),
                "threshold": self.meta.get("threshold"),
                "model": self.meta.get("model_used"),
                "model_weight": self.meta.get("model_weight"),
                "ela_quality": self.features.ela_quality,
                "host_quality": self.features.host_quality,
                "elapsed_s": self.meta.get("elapsed_s"),
                "channel_weights": CLASSICAL_WEIGHTS,
            },
            "regions": [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "kind_scores": {k: num(v) for k, v in r.kind_scores.items()},
                    "confidence": num(r.confidence),
                    "bbox": r.bbox,
                    "area_frac": num(r.area_frac),
                    "mean_prob": num(r.mean_prob),
                    "copy_source": r.copy_source,
                    "stats": {k: num(v) for k, v in r.stats.items()},
                    "evidence": r.evidence,
                    "attribution": {k: num(v) for k, v in r.attribution.items()},
                }
                for r in self.regions
            ],
            "order": {
                "sequence": self.order.sequence,
                "confidence": num(self.order.confidence),
                "decided_pairs": self.order.decided_pairs,
                "cycles_broken": self.order.cycles_broken,
                "edges": self.order.edges,
                "undecided": self.order.undecided,
            },
            "provenance": self.provenance.to_dict(),
            "pairs": [
                {"a": p.a, "b": p.b, "score": num(p.score),
                 "cues": {k: num(v) for k, v in p.cues.items()},
                 "available": p.available}
                for p in self.pairs
            ],
        }


def _load(source, cfg: Config | None = None) -> ImageBundle:
    if isinstance(source, ImageBundle):
        return source
    if isinstance(source, np.ndarray):
        return bundle_from_array(source)
    return load_image(source, max_side=(cfg.max_side if cfg else 1600))


def analyze(source, cfg: Config | None = None, use_model: bool = True) -> Trace:
    cfg = cfg or Config()
    t0 = time.perf_counter()

    bundle = _load(source, cfg)
    features = build_features(bundle, cfg)
    classical = classical_score(features)

    model = load_if_available(cfg) if use_model else None
    model_prob = model.predict(features, tile=cfg.tile) if model is not None else None
    prob = (blend_scores(model_prob, classical, cfg.model_weight)
            if model_prob is not None else classical)

    dct_cache = block_dct(features.bundle.gray)
    dq_q, dq_s = dq_scan(dct_cache)

    regions, mask = extract_regions(features, prob, cfg, dct_cache=dct_cache)
    for r in regions:
        classify_region(r, bundle.bgr)

    pairs = all_pair_evidence(regions, features, cfg)
    precedence = score_pairs(pairs, regions, features, cfg.resolved_precedence_path())
    order = order_from_evidence(pairs, cfg, {r.id: r.confidence for r in regions})

    attribute_channels(features, regions)
    for r in regions:
        build_evidence(r, features)

    edited_frac = float(mask.mean() / 255.0) if mask.size else 0.0
    prov = audit(bundle, (dq_q, dq_s), edit_area_frac=edited_frac)

    peak = float(prob.max()) if prob.size else 0.0
    manipulated = bool(len(regions) > 0 and (peak > cfg.mask_threshold or edited_frac > 0.005))

    meta = {
        "threshold": cfg.mask_threshold,
        "model_used": model is not None,
        "precedence_model": precedence is not None,
        "model_weight": cfg.model_weight if model is not None else 0.0,        "manipulated": manipulated,
        "peak_prob": peak,
        "edited_area_frac": edited_frac,
        "dq_global": (dq_q, dq_s),
        "elapsed_s": time.perf_counter() - t0,
        "sha256": hashlib.sha256(bundle.raw).hexdigest() if bundle.raw else "",
        "channels": list(CHANNELS),
        "config": {
            "mask_threshold": cfg.mask_threshold,
            "min_region_frac": cfg.min_region_frac,
            "order_weights": cfg.order_weights,
        },
    }
    return Trace(
        bundle=bundle, features=features, prob=prob, classical=classical,
        model_prob=model_prob, regions=regions, order=order, pairs=pairs,
        provenance=prov, mask=mask, meta=meta,
    )
