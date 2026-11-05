"""Procedural scenes and synthetic tampering with ground truth.

Two jobs:

1. produce demo material that looks like a photograph and carries *real*
   forensic traces (grain, demosaic-ish chroma noise, JPEG history), and
2. produce unlimited training pairs ``(image, mask)`` plus an edit list whose
   order is known, so the ordering stage can be scored honestly.

The generator is deliberately physical rather than adversarial: it does not
know what the detector looks for, it only simulates what an editor does
(paste, clone, inpaint, resize, feathered blend, re-save).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .io_utils import encode_jpeg


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fractal(rng: np.random.Generator, h: int, w: int, octaves: int = 5,
             base: int = 3, persistence: float = 0.55) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = base * 2 ** o
        gw = max(2, int(base * 2 ** o * w / max(h, 1)))
        grid = rng.random((gh + 1, gw + 1)).astype(np.float32)
        n = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * n
        total += amp
        amp *= persistence
    out /= max(total, 1e-6)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _palette(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """BGR palettes for day / dusk / overcast scenes."""
    mood = rng.integers(0, 3)
    if mood == 0:      # clear day
        sky_top = np.array([0.78, 0.50, 0.22])
        sky_bot = np.array([0.86, 0.76, 0.62])
        sun = np.array([0.95, 0.98, 1.00])
        ground = np.array([0.30, 0.36, 0.33])
    elif mood == 1:    # dusk
        sky_top = np.array([0.42, 0.24, 0.16])
        sky_bot = np.array([0.55, 0.52, 0.55])
        sun = np.array([0.55, 0.72, 1.00])
        ground = np.array([0.20, 0.22, 0.20])
    else:              # overcast
        sky_top = np.array([0.70, 0.68, 0.66])
        sky_bot = np.array([0.80, 0.80, 0.78])
        sun = np.array([0.90, 0.92, 0.92])
        ground = np.array([0.32, 0.34, 0.31])
    return {"sky_top": sky_top, "sky_bot": sky_bot, "sun": sun, "ground": ground}


# --------------------------------------------------------------------------
# scene rendering
# --------------------------------------------------------------------------
def render_scene(seed: int, size: tuple[int, int] = (384, 512),
                 supersample: int = 2, grain: float | None = None,
                 jpeg_quality: int | None = None) -> np.ndarray:
    """Render a textured landscape as a BGR uint8 image."""
    rng = np.random.default_rng(seed)
    h, w = size
    H, W = h * supersample, w * supersample
    pal = _palette(rng)

    yy = (np.linspace(0, 1, H, dtype=np.float32) ** 0.75)[:, None, None]
    sky = pal["sky_top"][None, None, :] * (1 - yy) + pal["sky_bot"][None, None, :] * yy
    img = np.repeat(sky, W, axis=1)

    # clouds
    clouds = _fractal(rng, H, W, octaves=5, base=2)
    cloud_a = np.clip((clouds - 0.48) * 2.4, 0, 1)[..., None] * 0.55
    img = img * (1 - cloud_a) + np.array([1.0, 1.0, 1.0])[None, None, :] * cloud_a

    # sun and glow
    cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.05, 0.3) * H
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / (H * H * 0.09)
    glow = np.exp(-r2)[..., None]
    img = img + (pal["sun"][None, None, :] - img) * (0.55 * glow)

    # mountain ridges
    horizon = int(H * rng.uniform(0.48, 0.62))
    for ridge in range(int(rng.integers(1, 3))):
        line = np.linspace(0, 1, W, dtype=np.float32)
        prof = _fractal(rng, 1, W, octaves=4, base=2)[0]
        y = horizon - (0.10 + 0.06 * ridge) * H * prof - 0.02 * H
        y = np.clip(y, 0, H - 1)
        m = (gy > y[None, :]).astype(np.float32)
        shade = 0.65 + 0.35 * _fractal(rng, H, W, octaves=4, base=4)
        col = (pal["ground"] * (0.55 + 0.08 * ridge))[None, None, :] * shade[..., None]
        haze = 0.22 + 0.18 * ridge
        col = col * (1 - haze) + pal["sky_bot"][None, None, :] * haze
        img = img * (1 - m[..., None]) + col * m[..., None]

    # ground
    ground_m = (gy > horizon).astype(np.float32)[..., None]
    tex = _fractal(rng, H, W, octaves=6, base=3)
    gcol = pal["ground"][None, None, :] * (0.6 + 0.8 * tex[..., None])
    depth = ((gy - horizon) / max(H - horizon, 1))[..., None]
    gcol = gcol * (0.8 + 0.35 * depth)
    img = img * (1 - ground_m) + gcol * ground_m

    # objects: trees, rocks, a couple of highlights
    for _ in range(int(rng.integers(6, 13))):
        ox = int(rng.uniform(0.05, 0.95) * W)
        oy = int(rng.uniform(horizon + 0.02 * H, 0.97 * H))
        rx = int(rng.uniform(0.012, 0.045) * W)
        ry = int(rx * rng.uniform(0.9, 2.2))
        col = pal["ground"] * rng.uniform(0.35, 0.75)
        if rng.random() < 0.45:   # tree
            cv2.rectangle(img, (ox - max(1, rx // 6), oy), (ox + max(1, rx // 6), oy + ry),
                          tuple(float(c) for c in col * 0.6), -1)
            cv2.ellipse(img, (ox, oy - ry // 3), (rx, int(ry * 0.9)), 0, 0, 360,
                        tuple(float(c) for c in col * 1.15), -1)
        else:                     # rock / bush
            cv2.ellipse(img, (ox, oy), (rx, ry), rng.uniform(0, 180), 0, 360,
                        tuple(float(c) for c in col), -1)
            cv2.ellipse(img, (ox - rx // 3, oy - ry // 3), (rx // 2, ry // 2), 0, 0, 360,
                        tuple(float(c) for c in col * 1.25), -1)

    # vignette + tone
    rr = ((gx - W / 2) ** 2 + (gy - H / 2) ** 2) / (0.5 * (W ** 2 + H ** 2))
    img = img * (1 - 0.22 * rr[..., None])
    img = np.clip(img, 0, 1) ** float(rng.uniform(0.92, 1.06))

    # grain, then demosaic-like chroma noise
    if grain is None:
        grain = float(rng.uniform(1.8, 4.2))
    img = img * 255.0
    img += rng.normal(0, grain, (H, W, 1)).astype(np.float32)
    chroma = rng.normal(0, grain * 0.55, (H, W, 3)).astype(np.float32)
    img += chroma - chroma.mean(axis=2, keepdims=True)

    img = np.clip(img, 0, 255).astype(np.uint8)
    if supersample > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if jpeg_quality is not None:
        img = encode_jpeg(img, jpeg_quality)
    return img


# --------------------------------------------------------------------------
# tampering
# --------------------------------------------------------------------------
@dataclass
class Edit:
    order: int
    kind: str
    bbox: tuple[int, int, int, int]           # x, y, w, h
    area_frac: float
    source: str = ""
    meta: dict = field(default_factory=dict)
    mask: np.ndarray | None = None


@dataclass
class SyntheticSample:
    bgr: np.ndarray
    mask: np.ndarray
    edits: list[Edit]
    meta: dict

    @property
    def order(self) -> list[int]:
        return [e.order for e in sorted(self.edits, key=lambda e: e.order)]


def _blob_mask(h: int, w: int, box: tuple[int, int, int, int],
               rng: np.random.Generator, roundness: float = 0.4) -> np.ndarray:
    x, y, bw, bh = box
    m = np.zeros((h, w), np.uint8)
    if rng.random() < roundness:
        cv2.ellipse(m, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2),
                    rng.uniform(0, 180), 0, 360, 255, -1)
    else:
        cv2.rectangle(m, (x, y), (x + bw, y + bh), 255, -1)
    return m


def _place_box(rng, h, w, fw, fh, occupied) -> tuple[int, int, int, int] | None:
    bw = max(24, int(w * fw))
    bh = max(24, int(h * fh))
    for _ in range(24):
        x = int(rng.uniform(0.02, 0.98) * w)
        y = int(rng.uniform(0.02, 0.98) * h)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        if x2 - x < 24 or y2 - y < 24:
            continue
        trial = np.zeros((h, w), np.uint8)
        trial[y:y2, x:x2] = 1
        if occupied is not None and (trial & occupied).sum() > 0.05 * trial.sum():
            continue
        return x, y, x2 - x, y2 - y
    return None


def make_tampered(seed: int, size: tuple[int, int] = (384, 512),
                  n_edits: int = 2, host_quality: int = 84,
                  final_quality: int = 92,
                  host: np.ndarray | None = None,
                  donor: np.ndarray | None = None) -> SyntheticSample:
    """Build a tampered image with known masks, kinds and edit order.

    ``host``/``donor`` may be supplied (e.g. real photographs) instead of
    procedurally rendered scenes. Everything downstream is identical.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    h, w = size
    if host is None:
        host = render_scene(seed, size=size)
    else:
        host = cv2.resize(host, (w, h), interpolation=cv2.INTER_AREA)
    if donor is None:
        donor = render_scene(seed + 104729, size=size)
    else:
        donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_AREA)
    host = encode_jpeg(host, host_quality)          # camera-side save
    donor = encode_jpeg(donor, host_quality + int(rng.integers(-4, 5)))

    canvas = host.astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    occupied = np.zeros((h, w), np.uint8)
    edits: list[Edit] = []

    kinds = ["splice", "copy_move", "inpaint"]
    weights = np.array([0.5, 0.3, 0.2])
    for i in range(n_edits):
        kind = str(rng.choice(kinds, p=weights / weights.sum()))
        fw = float(rng.uniform(0.14, 0.34))
        fh = float(rng.uniform(0.14, 0.34))
        box = _place_box(rng, h, w, fw, fh, occupied if occupied.any() else None)
        if box is None:
            continue
        x, y, bw, bh = box
        m = _blob_mask(h, w, box, rng)

        if kind == "splice":
            # crop a donor patch and resample it to the target size
            sx = int(rng.uniform(0, w - bw))
            sy = int(rng.uniform(0, h - bh))
            src = donor[sy:sy + bh, sx:sx + bw].astype(np.float32)
            if rng.random() < 0.6:
                src = cv2.resize(src, (max(16, int(bw * rng.uniform(0.7, 1.35))),
                                       max(16, int(bh * rng.uniform(0.7, 1.35)))),
                                 interpolation=cv2.INTER_CUBIC)
            src = cv2.resize(src, (bw, bh), interpolation=cv2.INTER_CUBIC)
            gain = 1.0 + rng.normal(0, 0.06, 3).astype(np.float32)
            offset = rng.normal(0, 4.0, 3).astype(np.float32)
            src = np.clip(src * gain[None, None, :] + offset[None, None, :], 0, 255)
            src += rng.normal(0, rng.uniform(1.0, 5.0), src.shape).astype(np.float32)
            region = canvas[y:y + bh, x:x + bw]
            mm = m[y:y + bh, x:x + bw]
            alpha = cv2.GaussianBlur(mm.astype(np.float32) / 255.0, (0, 0),
                                     rng.uniform(0.6, 2.6))
            canvas[y:y + bh, x:x + bw] = region * (1 - alpha[..., None]) + src * alpha[..., None]
            source = f"donor scene #{seed + 104729} crop {bw}x{bh} @ ({sx},{sy})"

        elif kind == "copy_move":
            # clone from the same image; rotate/scale to break block alignment
            sx = int(rng.uniform(0, w - bw))
            sy = int(rng.uniform(0, h - bh))
            if abs(sx - x) < bw and abs(sy - y) < bh:
                sx = (sx + w // 2) % max(1, w - bw)
            src = host[sy:sy + bh, sx:sx + bw].astype(np.float32)
            if rng.random() < 0.5:
                ang = rng.uniform(-7, 7)
                scale = rng.uniform(0.85, 1.15)
                M = cv2.getRotationMatrix2D((bw / 2, bh / 2), ang, scale)
                src = cv2.warpAffine(src, M, (bw, bh), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REFLECT)
            src = src + rng.normal(0, 1.2, src.shape).astype(np.float32)
            mm = m[y:y + bh, x:x + bw]
            alpha = cv2.GaussianBlur(mm.astype(np.float32) / 255.0, (0, 0),
                                     rng.uniform(0.6, 2.0))
            region = canvas[y:y + bh, x:x + bw]
            canvas[y:y + bh, x:x + bw] = region * (1 - alpha[..., None]) + src * alpha[..., None]
            source = f"clone from ({sx},{sy})"

        else:  # inpaint / removal
            src_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
            mm = cv2.dilate(m, np.ones((3, 3), np.uint8))
            filled = cv2.inpaint(src_u8, mm, 5, cv2.INPAINT_TELEA).astype(np.float32)
            alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 1.2)
            canvas = canvas * (1 - alpha[..., None]) + filled * alpha[..., None]
            source = "content-aware fill"

        mask = np.maximum(mask, m)
        occupied = np.maximum(occupied, m)
        edits.append(Edit(order=i, kind=kind, bbox=box,
                          area_frac=float(m.mean() / 255.0), source=source,
                          mask=m.copy(), meta={"pre_saves": 0}))

        # Multi-session workflow: an editor saves between operations. Anything
        # already on the canvas then accumulates another compression
        # generation, which is exactly the asymmetry the ordering stage reads.
        if i < n_edits - 1 and rng.random() < 0.9:
            mid_q = int(rng.integers(62, 77))
            canvas = encode_jpeg(np.clip(canvas, 0, 255).astype(np.uint8), mid_q).astype(np.float32)
            for e in edits:
                e.meta["pre_saves"] = int(e.meta.get("pre_saves", 0)) + 1
            edits[-1].meta["mid_quality"] = mid_q

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    out = encode_jpeg(out, final_quality)

    return SyntheticSample(
        bgr=out,
        mask=mask,
        edits=edits,
        meta={
            "seed": seed,
            "host_quality": int(host_quality),
            "final_quality": int(final_quality),
            "double_compressed": host_quality != final_quality,
            "size": [h, w],
        },
    )


def make_clean(seed: int, size: tuple[int, int] = (384, 512),
               quality: int = 92) -> np.ndarray:
    return render_scene(seed, size=size, jpeg_quality=quality)


def write_jpeg_with_exif(bgr: np.ndarray, path, quality: int = 92,
                         software: str | None = None,
                         make: str | None = None, model: str | None = None):
    """Save a JPEG with a minimal EXIF block (for provenance demos/tests)."""
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    if software:
        exif[305] = software
    if make:
        exif[271] = make
    if model:
        exif[272] = model
    exif[306] = "2024:03:11 09:14:22"
    img.save(path, quality=quality, exif=exif.tobytes())
