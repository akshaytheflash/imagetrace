"""Choose a mask threshold on a *training* split, never on the test split.

Caches raw model probabilities once, then sweeps thresholds so the sweep is
cheap. Reports pixel F1 and image-level MCC for each candidate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edittrace.config import Config
from edittrace.data import columbia_rows, coverage_rows
from edittrace.forensics.stack import build_features
from edittrace.io_utils import load_image
from edittrace.models.fusion import load_if_available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--dataset", default="columbia", choices=["coverage", "columbia"])
    ap.add_argument("--model", default="edittrace/weights/edittrace_real.pt")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=12)
    a = ap.parse_args()

    loader = columbia_rows if a.dataset == "columbia" else coverage_rows
    rows = loader(a.root, split="train")
    forged = [r for r in rows if r["tampered"]][: a.per_class]
    authentic = [r for r in rows if not r["tampered"]][: a.per_class]
    cfg = Config(max_side=a.max_side, model_path=a.model)
    model = load_if_available(cfg)
    if model is None:
        raise SystemExit("no model")

    cache = []
    for i, row in enumerate(forged + authentic):
        b = load_image(row["image"], max_side=a.max_side)
        st = build_features(b)
        p = model.predict(st)
        if row["mask"] is not None:
            gt = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (b.w, b.h), interpolation=cv2.INTER_NEAREST) > 127
        else:
            gt = np.zeros((b.h, b.w), bool)
        cache.append((p, gt, row["tampered"]))
        print(f"[{i + 1}/{len(forged) + len(authentic)}] {row['id']} "
              f"{'forged' if row['tampered'] else 'real'} p99={np.quantile(p, 0.99):.3f}")

    print("\nthr  pix_prec  pix_rec  pix_f1   pix_iou  img_mcc")
    best = None
    for t in np.arange(0.30, 0.86, 0.05):
        tp = fp = tn = fn = 0
        itp = ifp = itn = ifn = 0
        for p, gt, tampered in cache:
            pred = p > t
            tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum())
            tn += int((~pred & ~gt).sum()); fn += int((~pred & gt).sum())
            detected = bool(pred.mean() > 0.002)
            if tampered:
                itp += int(detected); ifn += int(not detected)
            else:
                ifp += int(detected); itn += int(not detected)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        iou = tp / max(tp + fp + fn, 1)
        den = np.sqrt(max((itp + ifp) * (itp + ifn) * (itn + ifp) * (itn + ifn), 1))
        mcc = (itp * itn - ifp * ifn) / den
        print(f"{t:4.2f}  {prec:8.3f}  {rec:7.3f}  {f1:7.3f}  {iou:8.3f}  {mcc:7.3f}")
        if best is None or f1 > best[1]:
            best = (float(t), f1, mcc)
    print(f"\nbest by pixel F1: threshold {best[0]:.2f}  F1 {best[1]:.3f}  image MCC {best[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
