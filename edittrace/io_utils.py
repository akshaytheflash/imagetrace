"""Image ingestion, metadata extraction and small numerical helpers."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ExifTags

# ITU-T T.81 Annex K luminance / chrominance tables.
STD_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

STD_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


# --------------------------------------------------------------------------
# robust statistics
# --------------------------------------------------------------------------
def robust_norm(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile stretch to [0, 1]; safe for constant maps."""
    x = np.asarray(x, dtype=np.float32)
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    if b - a < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Median/MAD z-score. Bounded to +/-8 to keep a single hot pixel sane."""
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < eps:
        mad = float(x.std()) + eps
    return np.clip((x - med) / mad, -8.0, 8.0)


def box_mean(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)
    return cv2.blur(np.asarray(x, dtype=np.float32), (k, k), borderType=cv2.BORDER_REFLECT)


def box_std(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = box_mean(x, k)
    m2 = box_mean(x * x, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def local_max(x: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(np.asarray(x, np.float32), np.ones((k, k), np.uint8))


def local_min(x: np.ndarray, k: int) -> np.ndarray:
    return cv2.erode(np.asarray(x, np.float32), np.ones((k, k), np.uint8))


# --------------------------------------------------------------------------
# JPEG helpers
# --------------------------------------------------------------------------
def quality_to_tables(quality: int) -> tuple[np.ndarray, np.ndarray]:
    q = int(np.clip(quality, 1, 100))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    f = np.clip((STD_LUMA * scale + 50.0) // 100.0, 1, 255)
    g = np.clip((STD_CHROMA * scale + 50.0) // 100.0, 1, 255)
    return f.astype(np.float32), g.astype(np.float32)


def estimate_quality(quant: dict[int, np.ndarray] | None) -> tuple[int | None, float]:
    """Match stored quantisation tables against the Annex K family.

    Returns ``(quality, normalised_error)``. A large error means a
    non-standard encoder or a re-saved file, which is itself evidence.
    """
    if not quant or 0 not in quant:
        return None, 1.0
    stored = np.asarray(quant[0], dtype=np.float32).reshape(8, 8)
    best_q, best_err = None, 1e9
    for q in range(1, 101):
        t, _ = quality_to_tables(q)
        err = float(np.abs(t - stored).mean())
        if err < best_err:
            best_q, best_err = q, err
    return best_q, float(np.clip(best_err / 12.0, 0.0, 1.0))


def extract_thumbnails(raw: bytes) -> list[np.ndarray]:
    """Pull every embedded JPEG (EXIF thumbnails, previews) out of a file."""
    thumbs: list[np.ndarray] = []
    starts = []
    i = 0
    while True:
        i = raw.find(b"\xff\xd8\xff", i)
        if i < 0:
            break
        starts.append(i)
        i += 3
    for s in starts[1:]:
        e = raw.find(b"\xff\xd9", s)
        if e < 0:
            continue
        chunk = raw[s:e + 2]
        if len(chunk) < 256:
            continue
        arr = cv2.imdecode(np.frombuffer(chunk, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None and min(arr.shape[:2]) >= 32:
            thumbs.append(arr)
        if len(thumbs) >= 3:
            break
    return thumbs


def thumbnail_similarity(full_bgr: np.ndarray, thumb_bgr: np.ndarray) -> float:
    """Normalised cross-correlation between the thumbnail and a downscale."""
    h, w = thumb_bgr.shape[:2]
    small = cv2.resize(full_bgr, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.clip((a * b).mean(), -1.0, 1.0))


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
def _read_tags(path: Path) -> tuple[dict[str, Any], dict[int, np.ndarray], bytes | None]:
    tags: dict[str, Any] = {}
    quant: dict[int, np.ndarray] = {}
    icc: bytes | None = None
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            for k, v in ex.items():
                name = ExifTags.TAGS.get(k, str(k))
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace").strip("\x00 ")
                tags[str(name)] = v
            try:
                sub = ex.get_ifd(0x8769)
                for k, v in sub.items():
                    tags[str(ExifTags.TAGS.get(k, k))] = str(v)[:200]
            except Exception:
                pass
            q = getattr(im, "quantization", None)
            if q:
                for kid, table in q.items():
                    quant[int(kid)] = np.asarray(table, dtype=np.float32).reshape(8, 8)
            icc = im.info.get("icc_profile")
            for key in ("xmp", "XML:com.adobe.xmp"):
                if im.info.get(key):
                    tags["_xmp"] = im.info[key].decode("utf-8", "replace") \
                        if isinstance(im.info[key], bytes) else str(im.info[key])
                    break
            tags["_format"] = im.format or path.suffix.lstrip(".").upper()
            tags["_mode"] = im.mode
    except Exception as exc:  # pragma: no cover - defensive
        tags["_error"] = str(exc)
    return tags, quant, icc


@dataclass
class ImageBundle:
    name: str
    bgr: np.ndarray                    # uint8 HxWx3
    rgb: np.ndarray                    # float32 HxWx3 in [0,1]
    gray: np.ndarray                   # float32 HxW in [0,1]
    tags: dict[str, Any] = field(default_factory=dict)
    quant: dict[int, np.ndarray] = field(default_factory=dict)
    icc: bytes | None = None
    raw: bytes = b""
    path: Path | None = None
    thumbnails: list[np.ndarray] = field(default_factory=list)

    @property
    def h(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def w(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def jpeg_quality(self) -> tuple[int | None, float]:
        return estimate_quality(self.quant)

    @property
    def filesize(self) -> int:
        return len(self.raw)

    def preview(self, max_side: int = 900) -> np.ndarray:
        h, w = self.bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s >= 1.0:
            return self.bgr.copy()
        return cv2.resize(self.bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def load_image(path: str | Path, max_side: int = 1600) -> ImageBundle:
    path = Path(path)
    raw = path.read_bytes()
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"cannot decode image: {path}")
    if max_side and max(arr.shape[:2]) > max_side:
        s = max_side / max(arr.shape[:2])
        arr = cv2.resize(arr, (int(arr.shape[1] * s), int(arr.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    tags, quant, icc = _read_tags(path)
    thumbs = extract_thumbnails(raw) if raw[:2] == b"\xff\xd8" else []
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return ImageBundle(
        name=path.name, bgr=arr, rgb=rgb, gray=gray, tags=tags, quant=quant,
        icc=icc, raw=raw, path=path, thumbnails=thumbs,
    )


def bundle_from_array(bgr: np.ndarray, name: str = "synthetic") -> ImageBundle:
    bgr = np.ascontiguousarray(bgr.astype(np.uint8))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return ImageBundle(name=name, bgr=bgr, rgb=rgb, gray=gray)


# --------------------------------------------------------------------------
# encoding helpers
# --------------------------------------------------------------------------
def imencode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def b64_png(bgr: np.ndarray) -> str:
    return base64.b64encode(imencode_png(bgr)).decode("ascii")


def encode_jpeg(bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def heatmap(x: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Normalise a scalar map and colourise it."""
    u8 = (robust_norm(x) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def overlay(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = heat if heat.shape[:2] == bgr.shape[:2] else cv2.resize(heat, (bgr.shape[1], bgr.shape[0]))
    return cv2.addWeighted(bgr, 1.0 - alpha, h, alpha, 0)


def human_size(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(n, 1024)))
    return f"{n / 1024 ** i:.1f} {units[i]}"
