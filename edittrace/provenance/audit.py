"""Container and metadata provenance audit.

Image forensics usually stops at the pixels. Most real edits, however, leave
paperwork: a software tag, a stale thumbnail, a re-save with non-standard
quantisation tables, an XMP history document. This module scores the
*provenance* of the file, independently of what the mask says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..io_utils import ImageBundle, thumbnail_similarity

EDITOR_HINTS = (
    "photoshop", "lightroom", "gimp", "affinity", "pixelmator", "pixlr",
    "snapseed", "canva", "paint.net", "capture one", "darktable", "rawtherapee",
    "midjourney", "stable diffusion", "dall", "firefly", "topaz", "facetune",
)

SEVERITY_PENALTY = {"high": 0.30, "medium": 0.16, "low": 0.07, "info": 0.0}


@dataclass
class Finding:
    id: str
    title: str
    detail: str
    severity: str
    value: str | None = None


@dataclass
class ProvenanceAudit:
    score: float
    verdict: str
    findings: list[Finding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "facts": self.facts,
            "findings": [f.__dict__ for f in self.findings],
        }


def _parse_dt(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(s.strip()[:19], fmt)
        except ValueError:
            continue
    return None


def audit(bundle: ImageBundle, dq_global: tuple[int, float] = (0, 0.0),
          edit_area_frac: float = 0.0) -> ProvenanceAudit:
    tags = bundle.tags or {}
    findings: list[Finding] = []
    fmt = str(tags.get("_format", "")).upper()
    q, q_err = bundle.jpeg_quality

    facts = {
        "format": fmt or "unknown",
        "dimensions": f"{bundle.w}x{bundle.h}",
        "filesize": bundle.filesize,
        "jpeg_quality_estimate": q,
        "quant_table_error": round(q_err, 3),
        "has_exif": bool(bundle.thumbnails) or any(
            k in tags for k in ("Make", "Model", "DateTime", "Software")),
        "make": tags.get("Make"),
        "model": tags.get("Model"),
        "software": tags.get("Software"),
        "embedded_thumbnails": len(bundle.thumbnails),
        "icc_profile_bytes": len(bundle.icc) if bundle.icc else 0,
        "xmp_present": bool(tags.get("_xmp")),
        "double_compression_score": round(float(dq_global[1]), 2),
        "double_compression_period": int(dq_global[0]) if dq_global[0] else None,
    }

    def add(fid, title, detail, severity, value=None):
        findings.append(Finding(fid, title, detail, severity, value))

    # -- editor fingerprints ------------------------------------------------
    software = str(tags.get("Software") or "").strip()
    if software:
        low = software.lower()
        if any(h in low for h in EDITOR_HINTS):
            add("software_tag", "Editing software recorded", 
                f'EXIF Software reads "{software}" - the file was written by an '
                "image editor, not a camera pipeline.", "high", software)
        else:
            add("software_tag_other", "Non-camera software tag",
                f'EXIF Software reads "{software}".', "medium", software)

    xmp = str(tags.get("_xmp") or "")
    if xmp:
        low = xmp.lower()
        hits = [h for h in ("crs:", "xmpmm", "stEvt", "photoshop", "history", "derivedfrom")
                if h.lower() in low]
        if hits:
            add("xmp_history", "XMP edit history present",
                "XMP packet contains editing-namespace records ("
                + ", ".join(hits[:4]) + "). The file documents its own retouching.",
                "high", ", ".join(hits[:4]))
        elif "xmp" in low:
            add("xmp_present", "XMP packet present", "XMP metadata is embedded but "
                "no edit-history namespace was found.", "info")

    # -- thumbnail integrity ------------------------------------------------
    if bundle.thumbnails:
        best = -1.0
        for th in bundle.thumbnails:
            ar_full = bundle.w / max(bundle.h, 1)
            ar_th = th.shape[1] / max(th.shape[0], 1)
            if abs(ar_full - ar_th) > 0.04:
                continue
            best = max(best, thumbnail_similarity(bundle.bgr, th))
        facts["thumbnail_similarity"] = round(best, 4) if best > -1 else None
        if best > -1 and best < 0.88:
            add("thumbnail_mismatch", "Embedded thumbnail does not match the image",
                f"EXIF thumbnail correlation is {best:.3f}. The preview was rendered "
                "from an earlier version of the file, so the pixels were changed after "
                "the metadata was written.", "high", f"NCC {best:.3f}")

    # -- metadata presence --------------------------------------------------
    camera_keys = [k for k in ("Make", "Model", "DateTimeOriginal", "ExposureTime",
                               "FNumber", "ISOSpeedRatings") if k in tags]
    if fmt == "JPEG" and not camera_keys and not software:
        add("metadata_stripped", "No camera metadata",
            "No Make/Model/exposure tags. Typical of screenshots, exports from "
            "editing tools, or deliberate metadata stripping.", "medium")
    elif fmt == "JPEG" and not tags.get("Make") and not tags.get("Model"):
        add("camera_identity_missing", "Camera identity missing",
            "Capture device could not be identified from EXIF.", "low")

    # -- compression history ------------------------------------------------
    if fmt == "JPEG":
        if q is not None and q_err > 0.30:
            add("nonstandard_quant", "Non-standard quantisation tables",
                f"Stored tables differ from the standard family (error {q_err:.2f}). "
                "Common after editing, transcoding or non-Adobe encoders.",
                "medium", f"q~{q}")
        if q is not None and q < 72:
            add("low_quality", "Heavy recompression",
                f"Estimated quality factor is {q}, low enough to destroy "
                "most camera-native noise structure.", "low", f"q={q}")
        if dq_global[1] >= 2.6 and dq_global[0] >= 2:
            add("double_compression", "Double-compression periodicity",
                f"Coefficient histogram shows a period of {dq_global[0]} "
                f"(score {dq_global[1]:.1f}) - the image was saved, edited and saved "
                "again.", "medium", f"period={dq_global[0]}")
        if bundle.icc is None:
            add("icc_missing", "No ICC profile", "Colour profile absent.", "info")
    elif fmt in ("PNG", "WEBP", "TIFF", "BMP"):
        add("lossless_container", "Non-camera container",
            f"{fmt} is an export/delivery format. There is no camera pipeline to "
            "anchor provenance.", "low", fmt)

    # -- byte-level container checks ---------------------------------------
    raw = bundle.raw
    if fmt == "JPEG" and raw:
        last_eoi = raw.rfind(b"\xff\xd9")
        if 0 <= last_eoi < len(raw) - 512:
            add("trailing_data", "Data after end-of-image",
                f"{len(raw) - last_eoi - 2} bytes follow the EOI marker; could be a "
                "second stream, an editor scratch pad, or benign padding.", "low",
                f"{len(raw) - last_eoi - 2} B")
        soi = [i for i in range(len(raw) - 2) if raw[i:i + 3] == b"\xff\xd8\xff"]
        if len(soi) > 1 and len(bundle.thumbnails) >= 2:
            add("multiple_streams", "Multiple embedded JPEG streams",
                f"{len(soi)} JPEG streams found in one file (thumbnail/preview plus "
                "main image).", "low", f"{len(soi)} streams")

    # -- timestamps ---------------------------------------------------------
    dt_orig = _parse_dt(tags.get("DateTimeOriginal"))
    dt_dig = _parse_dt(tags.get("DateTime"))
    if dt_orig and dt_dig and abs((dt_dig - dt_orig).total_seconds()) > 86400:
        add("timestamp_gap", "Capture and write timestamps diverge",
            f"DateTimeOriginal {dt_orig:%Y-%m-%d} vs DateTime {dt_dig:%Y-%m-%d}.",
            "medium")
    if bundle.path is not None:
        try:
            mtime = datetime.fromtimestamp(Path(bundle.path).stat().st_mtime)
            if dt_orig and mtime.year > dt_orig.year + 1:
                add("file_time", "File written long after capture",
                    f"Filesystem mtime {mtime:%Y-%m-%d} vs capture {dt_orig:%Y-%m-%d}.",
                    "info")
        except Exception:
            pass

    # -- score --------------------------------------------------------------
    score = 1.0
    for f in findings:
        score -= SEVERITY_PENALTY[f.severity]
    if edit_area_frac > 0.01:
        score -= min(0.18, 0.9 * edit_area_frac)
    score = float(np.clip(score, 0.02, 1.0))
    if score >= 0.78:
        verdict = "metadata consistent with a single capture"
    elif score >= 0.5:
        verdict = "provenance suspicious"
    else:
        verdict = "provenance inconsistent"
    return ProvenanceAudit(score=score, verdict=verdict, findings=findings, facts=facts)
