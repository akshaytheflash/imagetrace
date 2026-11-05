"""Dataset acquisition helpers.

The pipeline trains from procedural scenes out of the box. For real work you
want camera-native tampered pairs; this module wraps the two common routes so
the rest of the codebase only ever sees "a folder of images".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_SPLICING = "prajnar3/image-forgery-detection-dataset-splicing"

CATALOG = {
    "kaggle-splicing": {
        "slug": KAGGLE_SPLICING,
        "note": "101 real / 101 spliced wildlife photos with ground-truth masks.",
    },
    "casia2": {
        "url": "https://github.com/namtpham/casia2groundtruth",
        "note": "CASIA v2.0 with improved ground truth. Request access via the "
                "original authors; mirrors exist but check licensing.",
    },
    "coverage": {
        "url": "https://github.com/wenbihan/coverage",
        "note": "100 source/tampered pairs designed for copy-move and splicing.",
    },
    "columbia": {
        "url": "https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/",
        "note": "180 uncompressed splicing pairs; useful for compression-free tests.",
    },
    "defacto": {
        "url": "https://defactodataset.github.io/",
        "note": "Large, realistic splices with instance masks; heavy download.",
    },
}


def fetch(dataset: str, out: str | Path = "data/raw") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if dataset not in CATALOG:
        raise SystemExit(f"unknown dataset '{dataset}'. options: {', '.join(CATALOG)}")
    info = CATALOG[dataset]

    if dataset == "kaggle-splicing":
        if shutil.which("kaggle") is None:
            raise SystemExit(
                "kaggle CLI not found.\n"
                "  1. pip install kaggle\n"
                "  2. put your API token at %USERPROFILE%\\.kaggle\\kaggle.json\n"
                f"  3. edittrace fetch kaggle-splicing --out {out}\n"
                f"or download manually: https://www.kaggle.com/datasets/{KAGGLE_SPLICING}"
            )
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SPLICING,
               "-p", str(out), "--unzip"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return out

    raise SystemExit(
        f"{dataset}: {info['note']}\n"
        f"download from {info.get('url', '')}\n"
        f"then run: edittrace train --data {out}"
    )


def describe(path: str | Path) -> dict:
    """Quick inventory so `--data` failures are obvious before training."""
    root = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return {"root": str(root), "images": len(files),
            "subdirs": sorted({p.parent.name for p in files})[:12]}


def _benchmark_rows(root: Path, tp_name: str, au_name: str, split: str,
                    test_every: int, strip_suffix: str = "") -> list[dict]:
    tp_dir = root / tp_name
    au_dir = root / au_name
    records = json.loads((tp_dir / f"{tp_name}.json").read_text(encoding="utf-8"))
    rows = []
    for i, rec in enumerate(sorted(records, key=lambda r: r["forged_path"])):
        is_test = (i % test_every == 0)
        if split == "test" and not is_test:
            continue
        if split == "train" and is_test:
            continue
        stem = Path(rec["forged_path"]).stem
        if strip_suffix:
            stem = stem.replace(strip_suffix, "")
        rows.append({
            "id": stem,
            "image": tp_dir / rec["forged_path"].lstrip("./"),
            "mask": tp_dir / rec["mask_path"].lstrip("./"),
            "tampered": True,
            "split": "test" if is_test else "train",
        })
    for i, p in enumerate(sorted(au_dir.glob("*.png"))):
        is_test = (i % test_every == 0)
        if split == "test" and not is_test:
            continue
        if split == "train" and is_test:
            continue
        rows.append({"id": p.stem, "image": p, "mask": None, "tampered": False,
                     "split": "test" if is_test else "train"})
    return rows


def coverage_rows(root: str | Path, split: str = "train",
                  test_every: int = 3) -> list[dict]:
    """Rows for the COVERAGE layout (copy-move benchmark)."""
    return _benchmark_rows(Path(root), "coverage_tp", "coverage_au",
                           split, test_every, strip_suffix="t")


def columbia_rows(root: str | Path, split: str = "train",
                  test_every: int = 3) -> list[dict]:
    """Rows for the Columbia uncompressed splicing benchmark."""
    return _benchmark_rows(Path(root), "columbia_tp", "columbia_au",
                           split, test_every)
