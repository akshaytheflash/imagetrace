"""Quantitative evaluation on procedurally tampered scenes.

Reports the things that matter for this project: pixel localization, region
detection, edit-type accuracy, and - the hard one - whether the inferred order
matches the order the simulator actually applied.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .ordering.topo import kendall_tau
from .pipeline import analyze
from .synth import make_tampered


@dataclass
class SampleResult:
    seed: int
    iou: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    type_acc: float = 0.0
    tau: float = float("nan")
    order_conf: float = 0.0
    pair_acc: float = float("nan")
    pair_coverage: float = 0.0
    gt_edits: int = 0
    pred_regions: int = 0
    matched: int = 0
    elapsed: float = 0.0
    detail: dict = field(default_factory=dict)


def _match_regions(trace, sample) -> tuple[dict[int, int], dict[int, float]]:
    """pred region id -> gt edit index, plus overlap quality."""
    mapping: dict[int, int] = {}
    quality: dict[int, float] = {}
    for r in trace.regions:
        rm = r.mask > 0
        best, best_iou = -1, 0.0
        for e in sample.edits:
            if e.mask is None:
                continue
            em = e.mask > 0
            inter = float((rm & em).sum())
            union = float((rm | em).sum()) + 1e-6
            if inter / max(float(rm.sum()), 1.0) > 0.25:
                iou = inter / union
                if iou > best_iou:
                    best, best_iou = e.order, iou
        if best >= 0:
            mapping[r.id] = best
            quality[r.id] = best_iou
    return mapping, quality


def evaluate(n: int = 10, size: tuple[int, int] = (320, 400),
             use_model: bool = True, cfg: Config | None = None,
             seed0: int = 50_000, verbose: bool = True) -> dict:
    cfg = cfg or Config()
    rows: list[SampleResult] = []
    t0 = time.time()

    for k in range(n):
        seed = seed0 + k
        sample = make_tampered(seed, size=size, n_edits=2)
        ts = time.time()
        trace = analyze(sample.bgr, cfg, use_model=use_model)
        elapsed = time.time() - ts

        pred = trace.mask > 0
        gt = sample.mask > 0
        inter = float((pred & gt).sum())
        union = float((pred | gt).sum()) + 1e-6
        iou = inter / union
        f1 = 2 * inter / (float(pred.sum()) + float(gt.sum()) + 1e-6)

        mapping, quality = _match_regions(trace, sample)
        total_gt = len(sample.edits)
        # a predicted region counts as correct if it mostly covers one edit
        precision = len(mapping) / max(len(trace.regions), 1)
        recall = len(set(mapping.values())) / max(total_gt, 1)

        # edit type agreement on matched regions
        type_ok = 0
        for rid, gi in mapping.items():
            pred_kind = next(r.kind for r in trace.regions if r.id == rid)
            gt_kind = sample.edits[gi].kind
            if (pred_kind == "resampled_insert" and gt_kind == "splice") or \
               (pred_kind == "splice" and gt_kind in ("splice", "copy_move")):
                type_ok += 1
            elif pred_kind == gt_kind:
                type_ok += 1
        type_acc = type_ok / max(len(mapping), 1)

        tau = float("nan")
        if len(mapping) >= 2:
            gt_seq = [rid for rid in sorted(mapping, key=lambda r: mapping[r])]
            pred_seq = [rid for rid in trace.order.sequence if rid in mapping]
            tau = kendall_tau(gt_seq, pred_seq)

        pair_total = pair_decided = pair_correct = 0
        ids = sorted(mapping)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a not in trace.order.sequence or b not in trace.order.sequence:
                    continue
                gt_before = mapping[a] < mapping[b]
                pred_before = trace.order.sequence.index(a) < trace.order.sequence.index(b)
                pair_total += 1
                decided = any({e["before"], e["after"]} == {a, b} for e in trace.order.edges)
                if decided:
                    pair_decided += 1
                    pair_correct += int(pred_before == gt_before)
        pair_acc = pair_correct / pair_decided if pair_decided else float("nan")
        pair_cov = pair_decided / pair_total if pair_total else 0.0

        rows.append(SampleResult(
            seed=seed, iou=round(iou, 4), f1=round(f1, 4),
            precision=round(precision, 4), recall=round(recall, 4),
            type_acc=round(type_acc, 4), tau=tau,
            order_conf=round(trace.order.confidence, 4),
            pair_acc=round(pair_acc, 4) if not np.isnan(pair_acc) else float("nan"),
            pair_coverage=round(pair_cov, 4),
            gt_edits=total_gt, pred_regions=len(trace.regions),
            matched=len(mapping), elapsed=round(elapsed, 3),
            detail={
                "kinds": [e.kind for e in sample.edits],
                "pred_kinds": [r.kind for r in trace.regions],
                "order": trace.order.sequence,
                "gt_order": [mapping[r] for r in trace.order.sequence if r in mapping],
            },
        ))
        if verbose:
            print(f"[{k + 1:2d}/{n}] seed {seed} iou={iou:.3f} f1={f1:.3f} "
                  f"regions={len(trace.regions)}/{total_gt} tau={tau:+.2f} "
                  f"({elapsed:.2f}s)")

    def mean(key):
        vals = [getattr(r, key) for r in rows if not (isinstance(getattr(r, key), float)
                                                      and np.isnan(getattr(r, key)))]
        return float(np.mean(vals)) if vals else float("nan")

    taus = [r.tau for r in rows if not np.isnan(r.tau)]
    summary = {
        "n": n,
        "use_model": use_model,
        "mean_iou": round(mean("iou"), 4),
        "mean_f1": round(mean("f1"), 4),
        "mean_region_precision": round(mean("precision"), 4),
        "mean_region_recall": round(mean("recall"), 4),
        "mean_type_acc": round(mean("type_acc"), 4),
        "mean_order_tau": round(float(np.mean(taus)), 4) if taus else None,
        "order_samples": len(taus),
        "pair_accuracy": round(mean("pair_acc"), 4),
        "pair_coverage": round(mean("pair_coverage"), 4),
        "mean_order_conf": round(mean("order_conf"), 4),
        "mean_seconds": round(mean("elapsed"), 3),
        "total_seconds": round(time.time() - t0, 1),
        "samples": [asdict(r) for r in rows],
    }
    return summary


def to_markdown(summary: dict) -> str:
    lines = [
        f"| metric | value |", "| --- | --- |",
        f"| pixel IoU | {summary['mean_iou']:.3f} |",
        f"| pixel F1 | {summary['mean_f1']:.3f} |",
        f"| region precision | {summary['mean_region_precision']:.3f} |",
        f"| region recall | {summary['mean_region_recall']:.3f} |",
        f"| edit-type accuracy | {summary['mean_type_acc']:.3f} |",
        f"| order Kendall tau | {summary['mean_order_tau'] if summary['mean_order_tau'] is not None else 'n/a'} "
        f"(n={summary['order_samples']}) |",
        f"| pairwise accuracy (decided) | {summary['pair_accuracy']:.3f} |",
        f"| pairwise coverage | {summary['pair_coverage']:.3f} |",
        f"| order confidence | {summary['mean_order_conf']:.3f} |",
        f"| seconds / image | {summary['mean_seconds']:.2f} |",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=".out")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    s = evaluate(n=a.n, use_model=not a.no_model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "classical" if a.no_model else "fusion"
    (out / f"eval_{tag}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(to_markdown(s))
