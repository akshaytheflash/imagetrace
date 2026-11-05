"""Self-contained HTML report.

No external assets, no JavaScript frameworks: one file you can email to
someone who has to defend the finding. Images are base64 PNGs, the machine
readable trace is embedded as JSON, and every score carries the rule that
produced it.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..config import CHANNELS
from ..forensics.score import CLASSICAL_WEIGHTS
from ..io_utils import b64_png, human_size
from .evidence import build_evidence

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline import Trace

PALETTE = [
    (72, 176, 245),   # amber  (BGR)
    (235, 205, 70),   # cyan
    (215, 95, 235),   # magenta
    (120, 220, 140),  # green
    (245, 145, 90),   # blue
    (90, 90, 245),    # red
]

CHANNEL_HELP = {
    "ela": "Recompression error at the estimated file quality.",
    "noise_z": "Deviation of local sensor noise from the image median.",
    "noise_var": "Absolute local noise standard deviation.",
    "ghost_min": "Best-case JPEG ghost residual across a quality sweep.",
    "ghost_spread": "Spread of ghost residual across qualities.",
    "blockiness": "8x8 JPEG grid strength and phase.",
    "cfa_resid": "2x2 colour-filter / demosaic residual.",
    "resample": "Second-derivative periodicity from interpolation.",
    "seam": "Texture edges aligned with statistical discontinuity.",
    "hf_energy": "High-frequency energy.",
    "color_shift": "Local chroma deviation from a wide neighbourhood.",
    "edge_density": "Gradient density.",
}

SEVERITY_LABEL = {"high": "HIGH", "medium": "MED", "low": "LOW", "info": "INFO"}
KIND_LABEL = {
    "copy_move": "copy-move clone",
    "splice": "spliced insert",
    "resampled_insert": "resampled insert",
    "inpaint": "inpaint / removal",
    "unknown": "unclassified edit",
}


# --------------------------------------------------------------------------
# image rendering
# --------------------------------------------------------------------------
def _fit(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def annotate(bgr: np.ndarray, regions, max_w: int = 1180) -> np.ndarray:
    out = bgr.copy()
    tint = out.copy()
    for r in regions:
        col = PALETTE[(r.id - 1) % len(PALETTE)]
        tint[r.mask > 0] = col
    out = cv2.addWeighted(out, 0.66, tint, 0.34, 0)
    for r in regions:
        col = PALETTE[(r.id - 1) % len(PALETTE)]
        cnts, _ = cv2.findContours(r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2, cv2.LINE_AA)
        x, y, _, _ = r.bbox
        bx, by = max(16, x + 4), max(16, y + 4)
        cv2.circle(out, (bx, by), 14, (16, 18, 21), -1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 14, col, 2, cv2.LINE_AA)
        cv2.putText(out, str(r.id), (bx - 6 if r.id < 10 else bx - 10, by + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    return _fit(out, max_w)


def mask_panel(bgr: np.ndarray, regions, max_w: int = 640) -> np.ndarray:
    base = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) * 0.25).astype(np.uint8)
    panel = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for r in regions:
        col = PALETTE[(r.id - 1) % len(PALETTE)]
        panel[r.mask > 0] = col
    return _fit(panel, max_w)


def region_crop(bgr: np.ndarray, r, width: int = 260) -> np.ndarray:
    x, y, w, h = r.bbox
    pad = int(0.22 * max(w, h)) + 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    crop = bgr[y0:y1, x0:x1].copy()
    cnts, _ = cv2.findContours(r.mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, PALETTE[(r.id - 1) % len(PALETTE)], 2, cv2.LINE_AA)
    s = width / max(crop.shape[1], 1)
    return cv2.resize(crop, (width, int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)


def channel_thumb(stack, name: str, width: int = 300) -> np.ndarray:
    m = stack.maps[name]
    u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return _fit(col, width)


# --------------------------------------------------------------------------
# html fragments
# --------------------------------------------------------------------------
def _bar(value: float, colour: str = "amber") -> str:
    pct = max(2.0, min(100.0, value * 100))
    return f'<div class="bar"><span class="{colour}" style="width:{pct:.1f}%"></span></div>'


def _chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{html.escape(str(text))}</span>'


def _region_card(trace, r, imgs: dict) -> str:
    ev = build_evidence(r, trace.features)
    col = PALETTE[(r.id - 1) % len(PALETTE)]
    hexcol = "#%02x%02x%02x" % (col[2], col[1], col[0])
    attr_sorted = sorted(r.attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    attr_max = max([abs(v) for _, v in attr_sorted] + [1e-6])
    attr_html = "".join(
        f'<div class="attr-row"><span class="attr-name">{html.escape(k)}</span>'
        f'<div class="bar slim"><span class="attr" style="width:{abs(v) / attr_max * 100:.1f}%"></span></div>'
        f'<span class="attr-val">{v * 100:+.0f}%</span></div>'
        for k, v in attr_sorted
    )
    ev_html = "".join(
        f'<li><div class="ev-head"><span class="ev-clue">{html.escape(e["clue"])}</span>'
        f'<span class="ev-val">{html.escape(e["value"])}</span></div>'
        f'{_bar(e["strength"])}'
        f'<p class="ev-text">{html.escape(e["reading"])}</p></li>'
        for e in ev
    )
    src = ""
    if r.copy_source:
        src = (f'<p class="note">Cloned from approximately '
               f'({r.copy_source[0]}, {r.copy_source[1]}).</p>')
    kind = KIND_LABEL.get(r.kind, r.kind)
    return f"""
    <article class="region" style="--accent:{hexcol}">
      <header>
        <span class="rid">R{r.id}</span>
        <div>
          <h3>{html.escape(kind)}</h3>
          <p class="sub">bbox {r.bbox[0]},{r.bbox[1]} · {r.bbox[2]}×{r.bbox[3]} ·
             {r.area_frac * 100:.1f}% of frame</p>
        </div>
        <div class="conf">
          <strong>{r.confidence * 100:.0f}</strong><span>% confidence</span>
        </div>
      </header>
      <div class="region-body">
        <img src="data:image/png;base64,{imgs['crops'][r.id]}" alt="region {r.id} crop">
        <div class="region-right">
          <div class="scoreline">mask probability <b>{r.mean_prob:.2f}</b>
            <span class="dot">·</span> classifier <b>{r.kind_scores.get(r.kind, 0) * 100:.0f}%</b></div>
          <ul class="evidence">{ev_html}</ul>
          {src}
        </div>
      </div>
      <footer>
        <span class="foot-label">channel attribution</span>
        {attr_html}
      </footer>
    </article>"""


def _reachable(edges, src: int, dst: int) -> bool:
    adj: dict[int, list[int]] = {}
    for e in edges:
        adj.setdefault(e["before"], []).append(e["after"])
    stack, seen = [src], set()
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return False


def _order_html(trace) -> str:
    order = trace.order
    if not trace.regions:
        return '<p class="empty">No edit events to order.</p>'
    if len(trace.regions) == 1:
        return ('<p class="empty">A single region carries no pairwise ordering '
                'information; its placement in time rests on provenance alone.</p>')

    by_id = {r.id: r for r in trace.regions}
    steps = []
    for i, rid in enumerate(order.sequence):
        r = by_id[rid]
        col = PALETTE[(rid - 1) % len(PALETTE)]
        hexcol = "#%02x%02x%02x" % (col[2], col[1], col[0])
        steps.append(f"""
        <div class="step" style="--accent:{hexcol}">
          <span class="step-index">{i + 1:02d}</span>
          <div><b>R{rid}</b><span>{html.escape(KIND_LABEL.get(r.kind, r.kind))}</span></div>
        </div>""")
        if i < len(order.sequence) - 1:
            nxt = order.sequence[i + 1]
            edge = next((e for e in order.edges
                         if {e["before"], e["after"]} == {rid, nxt}), None)
            if edge:
                label = f"p={edge['score']:+.2f}"
            elif _reachable(order.edges, rid, nxt):
                label = "transitive"
            else:
                label = "no evidence"
            steps.append(f'<div class="arrow"><span>{label}</span><i></i></div>')

    cue_rows = "".join(
        f'<tr><td>R{e["before"]} → R{e["after"]}</td><td class="num">{e["score"]:+.2f}</td>'
        f'<td>{" · ".join(f"{k} {v:+.2f}" for k, v in e["cues"].items())}</td></tr>'
        for e in sorted(order.edges, key=lambda e: -abs(e["score"]))
    )
    undecided = "".join(
        f'<tr><td>R{u["a"]} ↔ R{u["b"]}</td><td class="num">{u["score"]:+.2f}</td>'
        f'<td>{html.escape(u["reason"])}</td></tr>'
        for u in order.undecided
    ) or '<tr><td colspan="3" class="empty">every pair was decided</td></tr>'

    return f"""
    <div class="timeline">{''.join(steps)}</div>
    <div class="order-meta">
      <div><span>sequence confidence</span><b>{order.confidence * 100:.0f}%</b></div>
      <div><span>decided pairs</span><b>{order.decided_pairs}</b></div>
      <div><span>conflicts resolved</span><b>{order.cycles_broken}</b></div>
    </div>
    <details class="table-details" open>
      <summary>Pairwise precedence</summary>
      <table class="data"><thead><tr><th>pair</th><th>score</th><th>cues (signed)</th></tr></thead>
      <tbody>{cue_rows or '<tr><td colspan="3" class="empty">none</td></tr>'}</tbody></table>
    </details>
    <details class="table-details">
      <summary>Undecided / conflicting pairs</summary>
      <table class="data"><thead><tr><th>pair</th><th>score</th><th>reason</th></tr></thead>
      <tbody>{undecided}</tbody></table>
    </details>"""


def _provenance_html(prov) -> str:
    rows = "".join(
        f'<li class="finding sev-{f.severity}">'
        f'<span class="sev">{SEVERITY_LABEL.get(f.severity, f.severity)}</span>'
        f'<div><b>{html.escape(f.title)}</b><p>{html.escape(f.detail)}</p></div>'
        f'{f"<code>{html.escape(str(f.value))}</code>" if f.value else ""}</li>'
        for f in prov.findings
    ) or '<li class="empty">No container-level anomalies.</li>'
    facts = prov.facts
    fact_rows = "".join(
        f"<tr><td>{html.escape(str(k).replace('_', ' '))}</td>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in facts.items() if v is not None
    )
    return f"""
    <div class="prov">
      <div class="gauge" style="--val:{prov.score * 100:.1f}">
        <div class="gauge-in"><b>{prov.score * 100:.0f}</b><span>/100</span></div>
      </div>
      <div class="prov-right">
        <p class="verdict">{html.escape(prov.verdict)}</p>
        <p class="note">This score is about the file, not the pixels: metadata, thumbnails,
        byte structure and quantisation tables. It is independent evidence.</p>
        <ul class="findings">{rows}</ul>
        <details class="table-details">
          <summary>Container facts</summary>
          <table class="data facts">{fact_rows}</table>
        </details>
      </div>
    </div>"""


def _channels_html(trace) -> str:
    cells = []
    for name in CHANNELS:
        w = CLASSICAL_WEIGHTS.get(name, 0.0)
        cells.append(f"""
        <figure class="channel">
          <img src="data:image/png;base64,{trace_maps(trace, name)}" alt="{name}">
          <figcaption><b>{name}</b><span class="w">w {w:.2f}</span>
          <p>{html.escape(CHANNEL_HELP.get(name, ''))}</p></figcaption>
        </figure>""")
    return "".join(cells)


_TRACE_CACHE: dict[int, dict] = {}


def trace_maps(trace, name: str) -> str:
    key = id(trace)
    cache = _TRACE_CACHE.setdefault(key, {})
    if name not in cache:
        cache[name] = b64_png(channel_thumb(trace.features, name))
    return cache[name]


# --------------------------------------------------------------------------
# main template
# --------------------------------------------------------------------------
def render_report(trace, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = trace.bundle.bgr
    regions = trace.regions
    imgs = {"crops": {}}
    imgs["hero"] = b64_png(annotate(bgr, regions))
    imgs["mask"] = b64_png(mask_panel(bgr, regions))
    for r in regions:
        imgs["crops"][r.id] = b64_png(region_crop(bgr, r))

    _TRACE_CACHE.pop(id(trace), None)
    channels = _channels_html(trace)

    max_prob = float(trace.prob.max()) if trace.prob.size else 0.0
    edited = float(trace.mask.mean()) / 255.0 if trace.mask.size else 0.0
    prov = trace.provenance
    model_state = "fusion U-Net + classical" if trace.meta.get("model_used") else "classical only"
    n = len(regions)
    headline = ("no edit regions survived thresholding" if n == 0 else
                f"{n} edit region{'s' if n != 1 else ''} localized")

    region_cards = "".join(_region_card(trace, r, imgs) for r in regions) or (
        '<p class="empty">The fused mask stayed below threshold everywhere. '
        'Absence of evidence is not proof of authenticity: see limitations.</p>')

    data_json = json.dumps(trace.to_dict(), default=float).replace("<", "\\u003c")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    sha = trace.meta.get("sha256", "")[:16]
    elapsed = trace.meta.get("elapsed_s", 0.0)

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EditTrace · {html.escape(trace.bundle.name)}</title>
<style>
:root {{
  --ink:#0a0b0d; --panel:#121417; --panel2:#171a1e; --line:#262a30;
  --text:#e8e5df; --muted:#8f959d; --dim:#5d646d;
  --amber:#f2b544; --red:#ff6b6b; --green:#5fd3a6; --blue:#6ea8fe;
}}
* {{ box-sizing:border-box; }}
html {{ background:var(--ink); }}
body {{
  margin:0; color:var(--text); background:
    radial-gradient(1200px 600px at 12% -10%, #1a1d22 0%, transparent 60%),
    radial-gradient(900px 500px at 95% 0%, #191712 0%, transparent 55%),
    var(--ink);
  font-family:"Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.55;
}}
.wrap {{ max-width:1240px; margin:0 auto; padding:34px 28px 90px; }}
code, .mono, .chip, .foot-label, th, .rid, .step-index, .sev, .w {{
  font-family:"JetBrains Mono", ui-monospace, "Cascadia Code", Consolas, monospace;
}}
a {{ color:var(--amber); }}
header.top {{
  display:flex; justify-content:space-between; align-items:flex-end;
  border-bottom:1px solid var(--line); padding-bottom:16px; margin-bottom:26px;
}}
.brand {{ font-family:"JetBrains Mono",monospace; letter-spacing:.22em; font-size:12px;
  text-transform:uppercase; color:var(--muted); }}
.brand b {{ color:var(--text); letter-spacing:.14em; }}
.meta {{ text-align:right; font-size:12px; color:var(--muted); }}
h1 {{ font-size:clamp(26px,3.4vw,40px); line-height:1.12; margin:6px 0 8px;
  letter-spacing:-.02em; font-weight:600; }}
h2 {{ font-size:13px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); font-weight:600; margin:54px 0 16px;
  display:flex; align-items:center; gap:12px; }}
h2:after {{ content:""; height:1px; flex:1; background:var(--line); }}
h3 {{ margin:0; font-size:16px; font-weight:600; letter-spacing:-.01em; }}
.sub {{ margin:2px 0 0; color:var(--muted); font-size:12px; }}
.note {{ color:var(--muted); font-size:12.5px; }}
.empty {{ color:var(--dim); font-style:italic; }}
.lede {{ color:var(--muted); max-width:70ch; margin:0 0 22px; font-size:15px; }}

.hero {{ display:grid; grid-template-columns:1.55fr 1fr; gap:22px; align-items:start; }}
.figure {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
  overflow:hidden; }}
.figure img {{ display:block; width:100%; height:auto; }}
.figure figcaption {{ padding:10px 14px; font-size:11.5px; color:var(--muted);
  display:flex; justify-content:space-between; gap:12px; border-top:1px solid var(--line); }}
.stats {{ display:grid; gap:12px; align-content:start; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; display:flex; justify-content:space-between; align-items:baseline; }}
.stat span {{ font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.stat b {{ font-size:24px; font-weight:600; letter-spacing:-.02em; }}
.stat small {{ color:var(--dim); font-size:12px; font-weight:400; }}
.verdict-row {{ display:flex; gap:10px; flex-wrap:wrap; margin:18px 0 4px; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line);
  background:var(--panel2); color:var(--muted); padding:4px 9px; border-radius:999px;
  font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; }}
.chip.hot {{ color:var(--red); border-color:#4a2528; background:#1d1416; }}
.chip.ok {{ color:var(--green); border-color:#1e3b32; background:#111a17; }}
.chip.warn {{ color:var(--amber); border-color:#43371f; background:#1b1710; }}

.region {{ border:1px solid var(--line); background:var(--panel); border-radius:14px;
  margin:18px 0; overflow:hidden; border-left:3px solid var(--accent); }}
.region header {{ display:flex; gap:14px; align-items:center; padding:16px 18px;
  border-bottom:1px solid var(--line); }}
.rid {{ width:42px; height:42px; border-radius:50%; border:1px solid var(--accent);
  color:var(--accent); display:grid; place-items:center; font-size:13px; flex:none; }}
.conf {{ margin-left:auto; text-align:right; }}
.conf strong {{ font-size:26px; font-weight:600; color:var(--accent); }}
.conf span {{ display:block; font-size:10px; letter-spacing:.12em; color:var(--dim);
  text-transform:uppercase; }}
.region-body {{ display:grid; grid-template-columns:270px 1fr; gap:20px; padding:16px 18px; }}
.region-body img {{ width:100%; border-radius:8px; border:1px solid var(--line); }}
.region-right {{ min-width:0; }}
.scoreline {{ font-size:12px; color:var(--muted); margin-bottom:10px; }}
.scoreline b {{ color:var(--text); font-weight:600; }}
.dot {{ padding:0 6px; color:var(--dim); }}
.evidence {{ list-style:none; margin:0; padding:0; }}
.evidence li {{ padding:9px 0; border-top:1px dashed #22262b; }}
.evidence li:first-child {{ border-top:none; padding-top:0; }}
.ev-head {{ display:flex; justify-content:space-between; gap:10px; font-size:12.5px; }}
.ev-clue {{ color:var(--text); font-weight:500; }}
.ev-val {{ color:var(--accent); font-family:"JetBrains Mono",monospace; font-size:11px; }}
.ev-text {{ margin:5px 0 0; color:var(--muted); font-size:12.5px; line-height:1.5; }}
.region footer {{ border-top:1px solid var(--line); padding:12px 18px 16px; }}
.foot-label {{ font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); }}
.attr-row {{ display:grid; grid-template-columns:120px 1fr 52px; gap:10px; align-items:center;
  margin-top:6px; font-size:11px; color:var(--muted); }}
.attr-name {{ font-family:"JetBrains Mono",monospace; }}
.attr-val {{ text-align:right; color:var(--text); font-family:"JetBrains Mono",monospace; }}
.bar {{ height:5px; background:#1c2025; border-radius:3px; overflow:hidden; margin-top:5px; }}
.bar.slim {{ margin-top:0; height:4px; }}
.bar span {{ display:block; height:100%; background:linear-gradient(90deg,#8a6320,var(--amber)); }}
.bar span.attr {{ background:linear-gradient(90deg,#2c4b64,var(--blue)); }}

.timeline {{ display:flex; align-items:stretch; gap:0; overflow-x:auto; padding:6px 2px 14px; }}
.step {{ background:var(--panel); border:1px solid var(--line);
  border-top:3px solid var(--accent); border-radius:10px; padding:12px 18px 12px 12px;
  display:flex; gap:12px; align-items:center; min-width:170px; }}
.step-index {{ color:var(--accent); font-size:11px; }}
.step b {{ display:block; font-size:14px; }}
.step span {{ font-size:11px; color:var(--muted); }}
.arrow {{ min-width:82px; display:flex; flex-direction:column; justify-content:center;
  align-items:center; gap:5px; color:var(--dim); font-family:"JetBrains Mono",monospace;
  font-size:9.5px; }}
.arrow i {{ display:block; width:100%; height:1px; background:var(--line); position:relative; }}
.arrow i:after {{ content:""; position:absolute; right:0; top:-3px;
  border-left:5px solid var(--line); border-top:3px solid transparent; border-bottom:3px solid transparent; }}
.order-meta {{ display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 14px; }}
.order-meta div {{ background:var(--panel); border:1px solid var(--line); border-radius:9px;
  padding:9px 13px; display:flex; gap:10px; align-items:baseline; }}
.order-meta span {{ font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }}
.order-meta b {{ font-family:"JetBrains Mono",monospace; }}
details.table-details {{ margin:10px 0; border:1px solid var(--line); border-radius:10px;
  background:var(--panel); padding:10px 14px; }}
details.table-details summary {{ cursor:pointer; font-size:12px; color:var(--muted);
  letter-spacing:.06em; }}
table.data {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:12px; }}
table.data th {{ text-align:left; color:var(--dim); font-weight:500; font-size:10px;
  letter-spacing:.1em; text-transform:uppercase; padding:6px 8px; border-bottom:1px solid var(--line); }}
table.data td {{ padding:7px 8px; border-bottom:1px solid #1b1e22; color:var(--muted); }}
table.data td.num {{ color:var(--text); font-family:"JetBrains Mono",monospace; }}
table.data td:first-child {{ color:var(--text); font-family:"JetBrains Mono",monospace; }}
.facts td:first-child {{ text-transform:capitalize; }}

.prov {{ display:grid; grid-template-columns:150px 1fr; gap:26px;
  background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:22px; }}
.gauge {{ width:132px; height:132px; border-radius:50%;
  background:conic-gradient(var(--amber) calc(var(--val) * 1%), #1e2227 0);
  display:grid; place-items:center; }}
.gauge-in {{ width:108px; height:108px; border-radius:50%; background:var(--panel);
  display:grid; place-items:center; align-content:center; }}
.gauge-in b {{ font-size:30px; font-weight:600; }}
.gauge-in span {{ font-size:10px; color:var(--dim); }}
.verdict {{ font-size:16px; margin:2px 0 4px; }}
.findings {{ list-style:none; padding:0; margin:14px 0 0; }}
.finding {{ display:grid; grid-template-columns:56px 1fr auto; gap:12px;
  padding:11px 0; border-top:1px solid #1d2126; align-items:start; }}
.finding b {{ font-size:13px; font-weight:600; }}
.finding p {{ margin:2px 0 0; color:var(--muted); font-size:12.5px; }}
.finding code {{ font-size:10.5px; color:var(--dim); background:#1a1e23;
  border:1px solid var(--line); padding:2px 6px; border-radius:5px; }}
.sev {{ font-size:9px; letter-spacing:.1em; padding:2px 5px; border-radius:4px;
  text-align:center; margin-top:2px; }}
.sev-high {{ color:var(--red); background:#201315; }}
.sev-medium {{ color:var(--amber); background:#1d1810; }}
.sev-low {{ color:var(--blue); background:#121a24; }}
.sev-info {{ color:var(--dim); background:#16191d; }}

.grid-channels {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.channel {{ margin:0; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; overflow:hidden; }}
.channel img {{ width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }}
.channel figcaption {{ padding:9px 11px 12px; }}
.channel b {{ font-family:"JetBrains Mono",monospace; font-size:11px; font-weight:500; }}
.channel .w {{ float:right; font-size:9.5px; color:var(--dim); }}
.channel p {{ margin:5px 0 0; font-size:11px; color:var(--muted); line-height:1.4; }}

.method {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.method .box {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; }}
.method h4 {{ margin:0 0 8px; font-size:12px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); }}
.method p, .method li {{ color:var(--muted); font-size:12.5px; }}
.method ul {{ margin:8px 0 0; padding-left:18px; }}
.method li {{ margin-bottom:6px; }}
footer.page {{ margin-top:60px; border-top:1px solid var(--line); padding-top:16px;
  display:flex; justify-content:space-between; color:var(--dim); font-size:11px;
  font-family:"JetBrains Mono",monospace; flex-wrap:wrap; gap:8px; }}
@media (max-width:960px) {{
  .hero, .region-body, .prov, .method {{ grid-template-columns:1fr; }}
  .grid-channels {{ grid-template-columns:repeat(2,1fr); }}
}}
</style></head>
<body><div class="wrap">
<header class="top">
  <div>
    <div class="brand">edit<b>trace</b> / forensic edit trace</div>
    <h1>{html.escape(headline)}</h1>
    <div class="mono" style="font-size:11.5px;color:var(--dim)">
      {html.escape(trace.bundle.name)} · {trace.bundle.w}×{trace.bundle.h} ·
      {human_size(trace.bundle.filesize)} · sha256:{html.escape(sha)}
    </div>
  </div>
  <div class="meta">
    <div>generated {generated}</div>
    <div>{elapsed:.2f}s · {model_state}</div>
    <div>edittrace 0.1.0</div>
  </div>
</header>

<div class="verdict-row">
  {_chip("manipulation " + ("detected" if trace.meta.get("manipulated") else "not established"),
         "hot" if trace.meta.get("manipulated") else "ok")}
  {_chip(f"edited area {edited * 100:.1f}%")}
  {_chip(f"peak mask {max_prob:.2f}")}
  {_chip(prov.verdict, "warn" if prov.score < 0.78 else "ok")}
  {_chip("jpeg q≈" + str(trace.features.host_quality) if trace.features.host_quality else "no jpeg tables")}
  {_chip(model_state)}
</div>

<p class="lede">The pipeline separates three questions that forgery detectors usually
collapse into one: <em>where</em> the pixels disagree with their surroundings,
<em>in what order</em> the disagreeing areas were written, and <em>whether the file's
paperwork</em> is consistent with the story the pixels tell. Numbers below are measured,
not generated; the method section states what would change them.</p>

<div class="hero">
  <figure class="figure">
    <img src="data:image/png;base64,{imgs['hero']}" alt="annotated analysis">
    <figcaption><span>localized regions and their boundaries</span>
      <span>{n} region{'s' if n != 1 else ''}</span></figcaption>
  </figure>
  <div class="stats">
    <div class="stat"><span>regions</span><b>{n}</b></div>
    <div class="stat"><span>edited area</span><b>{edited * 100:.1f}<small>%</small></b></div>
    <div class="stat"><span>peak mask</span><b>{max_prob:.2f}</b></div>
    <div class="stat"><span>order confidence</span>
      <b>{trace.order.confidence * 100:.0f}<small>%</small></b></div>
    <div class="stat"><span>provenance</span><b>{prov.score * 100:.0f}<small>/100</small></b></div>
    <figure class="figure" style="margin:0">
      <img src="data:image/png;base64,{imgs['mask']}" alt="mask">
      <figcaption><span>fused mask</span><span>threshold {trace.meta.get('threshold', 0.52):.2f}</span></figcaption>
    </figure>
  </div>
</div>

<h2>Edit order</h2>
{_order_html(trace)}

<h2>Region dossiers</h2>
{region_cards}

<h2>Provenance audit</h2>
{_provenance_html(prov)}

<h2>Forensic channels</h2>
<p class="note">Every channel is computed at full resolution, robustly normalised, then
fused. Attribution in the dossiers is measured by ablating one channel at a time and
re-scoring the region, so it reflects this image rather than a global prior.</p>
<div class="grid-channels">{channels}</div>

<h2>Method and limits</h2>
<div class="method">
  <div class="box">
    <h4>How the inference is built</h4>
    <ul>
      <li><b>Localization</b> - 12 hand-built forensic channels plus a 1.8M-parameter
          U-Net trained on synthetically tampered scenes; the output is blended with a
          transparent classical score ({trace.meta.get('model_weight', 0.62):.2f} model weight).</li>
      <li><b>Typing</b> - clone evidence from masked template matching, resampling
          periodicity, noise-floor direction, and JPEG ghost disagreement.</li>
      <li><b>Ordering</b> - four pairwise cues (compression lineage, seam truncation,
          seam blending, noise floor) feed a cycle-safe topological sort. Conflicts are
          reported, never silently averaged.</li>
      <li><b>Provenance</b> - EXIF/ICC/XMP inspection, thumbnail correlation, quantisation
          table matching, byte-structure checks.</li>
    </ul>
  </div>
  <div class="box">
    <h4>What can invalidate this</h4>
    <ul>
      <li>Ordering is only as good as trace <em>preservation</em>. A final lossless save,
          aggressive denoising, or screenshotting destroys the compression lineage cue.</li>
      <li>Copy-move edits inherit the host's noise, so noise-based localization is blind
          to them; only texture matching catches those.</li>
      <li>A clean image can be flagged if strong vignetting or intentional grain mimics
          foreign noise. The model is trained on procedural scenes and will transfer
          imperfectly to camera-native files.</li>
      <li>Provenance findings describe the container. They do not prove intent, and a
          channel editor can produce a perfectly consistent file.</li>
    </ul>
  </div>
</div>

<footer class="page">
  <span>edittrace 0.1.0 · deterministic given the same input and checkpoint</span>
  <span>machine-readable trace embedded as #trace-data</span>
</footer>
</div>
<script id="trace-data" type="application/json">{data_json}</script>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
