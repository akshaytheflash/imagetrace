"""Train the learned precedence model on synthetic edit pairs.

Uses ground-truth masks so the model learns the *ordering* signal rather than
the detector's mistakes; at inference it is applied to predicted regions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..forensics.dct import block_dct, dq_scan
from ..forensics.stack import build_features
from ..io_utils import bundle_from_array
from ..regions.classify import classify_region
from ..regions.extract import describe_region
from ..synth import make_tampered
from .cues import pair_evidence
from .features import FEATURE_NAMES, fit_logistic, load, pair_vector


def build_pairs(scenes: int = 160, size: tuple[int, int] = (320, 400),
                seed0: int = 800_000, verbose: bool = True):
    cfg = Config()
    X: list[np.ndarray] = []
    y: list[float] = []
    meta = []
    for k in range(scenes):
        seed = seed0 + k
        sample = make_tampered(seed, size=size, n_edits=2)
        if len(sample.edits) < 2:
            continue
        stack = build_features(bundle_from_array(sample.bgr))
        dct = block_dct(stack.bundle.gray)
        gq = dq_scan(dct)
        regions = []
        for i, e in enumerate(sample.edits):
            r = describe_region(stack, e.mask, i + 1, dct, gq, mean_prob=1.0)
            classify_region(r, sample.bgr)
            regions.append(r)
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                ev = pair_evidence(regions[i], regions[j], stack, cfg, gq)
                f = pair_vector(ev, regions[i], regions[j], stack)
                X.append(f)
                y.append(1.0)
                X.append(-f)
                y.append(0.0)
                meta.append({"seed": seed, "a": i, "b": j,
                             "kinds": [regions[i].kind, regions[j].kind]})
        if verbose and (k + 1) % 40 == 0:
            print(f"  {k + 1}/{scenes} scenes, {len(X)} directed pairs")
    return np.stack(X), np.asarray(y, np.float32), meta


def train_order(out: str | Path = "edittrace/weights/precedence.npz",
                scenes: int = 160, seed0: int = 800_000) -> dict:
    print(f"building {scenes} synthetic scenes ...")
    X, y, meta = build_pairs(scenes=scenes, seed0=seed0)
    print(f"dataset: {X.shape[0]} directed pairs, {X.shape[1]} features")
    model = fit_logistic(X, y)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    train_acc = float(np.mean((((X - model.mean) / model.scale) @ model.coef > 0) == (y > 0.5)))
    order = np.argsort(-np.abs(model.coef))
    top = [(FEATURE_NAMES[i], round(float(model.coef[i]), 3)) for i in order[:8]]
    report = {"scenes": scenes, "pairs": int(len(y) // 2), "train_acc": round(train_acc, 4),
              "top_features": top, "features": FEATURE_NAMES}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"train accuracy {train_acc:.3f}")
    print("top features:", top)
    print(f"saved -> {out}")
    return report


def score_pairs(pairs, regions, stack, path=None):
    """Attach learned margins to pair evidence (in place)."""
    from .features import load as load_model
    model = load_model(path) if path else None
    if model is None:
        return None
    by_id = {r.id: r for r in regions}
    for ev in pairs:
        a, b = by_id[ev.a], by_id[ev.b]
        f = pair_vector(ev, a, b, stack)
        m = model.margin(f)
        ev.cues["learned_precedence"] = float(np.tanh(m / 2.0))
        ev.score = float(np.clip(0.78 * np.tanh(m / 2.0) + 0.22 * ev.score, -1, 1))
    return model


if __name__ == "__main__":  # pragma: no cover
    train_order()
