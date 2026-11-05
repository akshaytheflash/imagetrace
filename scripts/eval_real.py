"""Evaluate EditTrace on a real tampering benchmark with ground-truth masks.

Default target: COVERAGE (100 forged / 100 authentic, 832x832) laid out as
    coverage_tp/forged/*.png, coverage_tp/gt/*forged.png, coverage_au/*.png

Reports both pixel-level and image-level confusion matrices, plus per-image
IoU. Ordering cannot be scored here because COVERAGE has no edit-order ground
truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edittrace.config import Config
from edittrace.data import columbia_rows, coverage_rows
from edittrace.pipeline import analyze


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4), "accuracy": round(acc, 4),
        "f1": round(f1, 4), "iou": round(iou, 4), "mcc": round(mcc, 4),
    }


def load_coverage(root: Path, split: str = "test", test_every: int = 3) -> list[dict]:
    return coverage_rows(root, split=split, test_every=test_every)


def load_rows(dataset: str, root: Path, split: str = "test") -> list[dict]:
    fn = columbia_rows if dataset == "columbia" else coverage_rows
    return fn(root, split=split)


def run(rows: list[dict], cfg: Config, use_model: bool = True,
        limit: int | None = None, dataset: str = "COVERAGE") -> dict:
    if limit:
        rows = rows[:limit]
    tp = fp = tn = fn = 0
    img_tp = img_fp = img_tn = img_fn = 0
    ious: list[float] = []
    fp_areas: list[float] = []
    per_image = []
    for i, row in enumerate(rows):
        trace = analyze(row["image"], cfg, use_model=use_model)
        pred = trace.mask > 0
        if row["mask"] is not None:
            gt = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            gt = cv2.resize(gt, (trace.bundle.w, trace.bundle.h),
                            interpolation=cv2.INTER_NEAREST) > 127
        else:
            gt = np.zeros_like(pred)

        a_tp = int((pred & gt).sum()); a_fp = int((pred & ~gt).sum())
        a_tn = int((~pred & ~gt).sum()); a_fn = int((~pred & gt).sum())
        tp += a_tp; fp += a_fp; tn += a_tn; fn += a_fn

        detected = bool(trace.meta.get("manipulated"))
        if row["tampered"]:
            img_tp += int(detected); img_fn += int(not detected)
            iou = a_tp / max(a_tp + a_fp + a_fn, 1)
            ious.append(iou)
            outcome = "TP" if detected else "FN"
        else:
            img_fp += int(detected); img_tn += int(not detected)
            fp_areas.append(float(pred.mean()))
            outcome = "FP" if detected else "TN"
        per_image.append({
            "id": row["id"], "tampered": row["tampered"], "outcome": outcome,
            "iou": round(ious[-1], 4) if row["tampered"] else None,
            "pred_area": round(float(pred.mean()), 4),
            "gt_area": round(float(gt.mean()), 4),
            "regions": len(trace.regions),
            "provenance": round(trace.provenance.score, 3),
            "seconds": round(trace.meta["elapsed_s"], 2),
        })
        print(f"[{i + 1}/{len(rows)}] {row['id']:>4} {'forged' if row['tampered'] else 'real  '} "
              f"{outcome}  iou={per_image[-1]['iou']}  pred_area={per_image[-1]['pred_area']}  "
              f"({trace.meta['elapsed_s']:.1f}s)")

    pixel = _metrics(tp, fp, tn, fn)
    image = _metrics(img_tp, img_fp, img_tn, img_fn)
    return {
        "dataset": dataset,        "source": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR (datasets/test/coverage_*.zip)",
        "use_model": use_model,
        "max_side": cfg.max_side,
        "threshold": cfg.mask_threshold,
        "n_tampered": sum(1 for r in rows if r["tampered"]),
        "n_authentic": sum(1 for r in rows if not r["tampered"]),
        "pixel": pixel,
        "image": image,
        "mean_iou_tampered": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou_tampered": round(float(np.median(ious)), 4) if ious else None,
        "mean_pred_area_authentic": round(float(np.mean(fp_areas)), 4) if fp_areas else None,
        "per_image": per_image,
    }


def to_markdown(r: dict) -> str:
    p, im = r["pixel"], r["image"]
    return "\n".join([
        f"**{r['dataset']}** — {r['n_tampered']} forged / {r['n_authentic']} authentic, "
        f"max side {r['max_side']}px, threshold {r['threshold']}",
        "",
        "| pixel | value | | image-level | value |",
        "| --- | --- | --- | --- | --- |",
        f"| TP | {p['tp']:,} | | TP | {im['tp']} |",
        f"| FP | {p['fp']:,} | | FP | {im['fp']} |",
        f"| TN | {p['tn']:,} | | TN | {im['tn']} |",
        f"| FN | {p['fn']:,} | | FN | {im['fn']} |",
        f"| precision | {p['precision']:.3f} | | precision | {im['precision']:.3f} |",
        f"| recall | {p['recall']:.3f} | | recall | {im['recall']:.3f} |",
        f"| specificity | {p['specificity']:.3f} | | specificity | {im['specificity']:.3f} |",
        f"| F1 | {p['f1']:.3f} | | F1 | {im['f1']:.3f} |",
        f"| IoU | {p['iou']:.3f} | | accuracy | {im['accuracy']:.3f} |",
        f"| MCC | {p['mcc']:.3f} | | MCC | {im['mcc']:.3f} |",
        "",
        f"Mean per-image IoU (forged): {r['mean_iou_tampered']} · "
        f"median: {r['median_iou_tampered']} · "
        f"mean predicted area on authentic: {r['mean_pred_area_authentic']}",
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--dataset", default="coverage", choices=["coverage", "columbia"])
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.58)
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-weight", type=float, default=0.8)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--out", default=".out/real_eval.json")
    a = ap.parse_args()
    cfg = Config(max_side=a.max_side, mask_threshold=a.threshold,
                 model_weight=a.model_weight)
    if a.model:
        cfg.model_path = a.model
    rows = load_rows(a.dataset, Path(a.root), split=a.split)
    res = run(rows, cfg, use_model=not a.no_model, limit=a.limit,
              dataset=a.dataset.upper())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + to_markdown(res))
