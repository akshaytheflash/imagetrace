"""Command line interface."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .pipeline import analyze
from .synth import make_tampered, write_jpeg_with_exif
from .io_utils import b64_png


def _json(trace) -> str:
    return json.dumps(trace.to_dict(), indent=2, default=float)


def cmd_analyze(args) -> int:
    cfg = Config(mask_threshold=args.threshold)
    if args.model:
        cfg.model_path = args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for p in args.images:
        path = Path(p)
        if path.is_dir():
            paths.extend(sorted(q for q in path.iterdir()
                                if q.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")))
        else:
            paths.append(path)
    if not paths:
        print("no images found")
        return 2

    for path in paths:
        trace = analyze(path, cfg, use_model=not args.no_model)
        stem = path.stem
        report = trace.render_html(out / f"{stem}.report.html")
        (out / f"{stem}.json").write_text(_json(trace), encoding="utf-8")
        n = len(trace.regions)
        area = trace.mask.mean() / 255.0 if trace.mask.size else 0.0
        print(f"{path.name}: {n} region(s), {area * 100:.1f}% area, "
              f"peak {trace.meta['peak_prob']:.2f}, provenance {trace.provenance.score:.2f}, "
              f"order conf {trace.order.confidence:.2f} -> {report}")
        for r in trace.regions:
            print(f"    R{r.id} {r.kind:<17} conf {r.confidence:.2f} "
                  f"bbox {r.bbox} top clue: "
                  f"{(r.evidence[0]['clue'] if r.evidence else 'none')}")
    return 0


def _overview(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
              width: int = 420) -> np.ndarray:
    """Grid: original+GT, original+prediction, prediction mask."""
    panels = []
    for name, gt_img, pred_img, mask_img in rows:
        ims = []
        for title, img in (("ground truth", gt_img), ("inferred", pred_img), ("mask", mask_img)):
            h, w = img.shape[:2]
            im = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((im.shape[0] + 26, im.shape[1], 3), np.uint8)
            canvas[26:] = im
            cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            ims.append(canvas)
        panels.append(np.hstack(ims))
    return np.vstack(panels)


def cmd_demo(args) -> int:
    from .regions.extract import Region  # noqa: F401  (documentation of shape)
    from .explain.report import PALETTE, annotate

    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    cfg = Config(mask_threshold=args.threshold)

    rows = []
    summary = []
    for i in range(args.n):
        seed = args.seed + i
        sample = make_tampered(seed, size=(384, 512), n_edits=args.edits)
        path = out / "samples" / f"sample_{i:02d}.jpg"
        if i == 0 and args.metadata:
            write_jpeg_with_exif(sample.bgr, path, quality=92,
                                 software="Adobe Photoshop 25.0 (Windows)",
                                 make="NIKON CORPORATION", model="NIKON Z6_2")
        else:
            cv2.imwrite(str(path), sample.bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        trace = analyze(path, cfg, use_model=not args.no_model)
        trace.render_html(out / f"sample_{i:02d}.report.html")
        (out / f"sample_{i:02d}.json").write_text(_json(trace), encoding="utf-8")

        gt = {"seed": seed, "edits": [
            {"order": e.order, "kind": e.kind, "bbox": e.bbox,
             "area_frac": round(e.area_frac, 4), "source": e.source}
            for e in sample.edits]}
        (out / f"sample_{i:02d}.truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")

        # GT visualisation
        gt_regions = []
        for e in sample.edits:
            class _G:
                pass
            g = _G()
            g.mask = e.mask
            g.bbox = e.bbox
            g.id = e.order + 1
            gt_regions.append(g)
        gt_img = annotate(sample.bgr, gt_regions, max_w=512)
        pred_img = annotate(sample.bgr, trace.regions, max_w=512)
        mask_vis = cv2.cvtColor(trace.mask, cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(trace.mask, (512, 384), interpolation=cv2.INTER_NEAREST)
        rows.append((path.name, gt_img, pred_img, cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)))

        summary.append({
            "sample": i, "seed": seed,
            "gt_kinds": [e.kind for e in sample.edits],
            "pred_kinds": [r.kind for r in trace.regions],
            "gt_edits": len(sample.edits), "pred_regions": len(trace.regions),
            "edited_area_pct": round(trace.mask.mean() / 255 * 100, 2),
            "order_sequence": trace.order.sequence,
            "order_confidence": round(trace.order.confidence, 3),
            "provenance_score": round(trace.provenance.score, 3),
        })
        print(f"[{i + 1}/{args.n}] {path.name}: {len(trace.regions)} regions, "
              f"order conf {trace.order.confidence:.2f}, prov {trace.provenance.score:.2f}")

    if rows:
        cv2.imwrite(str(out / "overview.png"), _overview(rows))
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndemo written to {out.resolve()}")
    print(f"open {out / 'sample_00.report.html'}")
    return 0


def cmd_synth(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        seed = args.seed + i
        sample = make_tampered(seed, size=(args.height, args.width), n_edits=args.edits)
        cv2.imwrite(str(out / f"tampered_{i:02d}.jpg"), sample.bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f"mask_{i:02d}.png"), sample.mask)
        truth = {"seed": seed, "edits": [
            {"order": e.order, "kind": e.kind, "bbox": e.bbox} for e in sample.edits]}
        (out / f"truth_{i:02d}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {args.n} tampered scenes to {out.resolve()}")
    return 0


def cmd_train(args) -> int:
    from .models.train import train
    res = train(out=args.out, steps=args.steps, batch=args.batch, size=args.size,
                lr=args.lr, data=args.data, width=args.width)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
    return 0


def cmd_train_order(args) -> int:
    from .ordering.train_order import train_order
    report = train_order(out=args.out, scenes=args.scenes, seed0=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))
    return 0


def cmd_fetch(args) -> int:
    from .data import describe, fetch
    out = fetch(args.dataset, args.out)
    print(describe(out))
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, to_markdown
    cfg = Config(mask_threshold=args.threshold)
    if args.model:
        cfg.model_path = args.model
    s = evaluate(n=args.n, use_model=not args.no_model, cfg=cfg, verbose=not args.quiet)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "classical" if args.no_model else "fusion"
    (out / f"eval_{tag}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    md = to_markdown(s)
    (out / f"eval_{tag}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edittrace",
                                description="Localize image edits, infer their order, audit provenance.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze images or a directory")
    a.add_argument("images", nargs="+")
    a.add_argument("--out", default=".out")
    a.add_argument("--no-model", action="store_true", help="classical pipeline only")
    a.add_argument("--model", default=None, help="path to a checkpoint")
    a.add_argument("--threshold", type=float, default=0.58)
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("demo", help="generate tampered samples, analyze, write reports")
    d.add_argument("--out", default=".out/demo")
    d.add_argument("--n", type=int, default=3)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--edits", type=int, default=2)
    d.add_argument("--threshold", type=float, default=0.58)
    d.add_argument("--no-model", action="store_true")
    d.add_argument("--metadata", action="store_true", default=True,
                   help="embed editor EXIF in the first sample")
    d.add_argument("--clean", action="store_true")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("synth", help="export synthetic tampered scenes with ground truth")
    s.add_argument("--out", default=".out/synth")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=384)
    s.add_argument("--edits", type=int, default=2)
    s.set_defaults(func=cmd_synth)

    t = sub.add_parser("train", help="train the fusion U-Net")
    t.add_argument("--out", default="edittrace/weights/edittrace_unet.pt")
    t.add_argument("--steps", type=int, default=420)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--size", type=int, default=256)
    t.add_argument("--lr", type=float, default=2.2e-4)
    t.add_argument("--width", type=int, default=24)
    t.add_argument("--data", default=None, help="folder of real photographs")
    t.set_defaults(func=cmd_train)

    to = sub.add_parser("train-order", help="train the learned precedence model")
    to.add_argument("--out", default="edittrace/weights/precedence.npz")
    to.add_argument("--scenes", type=int, default=160)
    to.add_argument("--seed", type=int, default=800_000)
    to.set_defaults(func=cmd_train_order)

    e = sub.add_parser("eval", help="evaluate on held-out synthetic scenes")
    e.add_argument("--n", type=int, default=10)
    e.add_argument("--out", default=".out")
    e.add_argument("--no-model", action="store_true")
    e.add_argument("--model", default=None)
    e.add_argument("--threshold", type=float, default=0.58)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_eval)

    f = sub.add_parser("fetch", help="download a public tampering dataset")
    f.add_argument("dataset", help="kaggle-splicing | casia2 | coverage | columbia | defacto")
    f.add_argument("--out", default="data/raw")
    f.set_defaults(func=cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
