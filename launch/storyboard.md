# EditTrace — launch film (64 s)

**File:** `launch/edittrace-launch.mp4` · 1920×1080 · 30 fps · H.264 / yuv420p / faststart
**Poster:** `launch/poster.png` · 1920×1080
**Composition:** `launch/build/comp.html` (onetake, one HTML file, every value a pure function of `t`)
**Sound:** `launch/build/sfx.wav` (synthesised from the film's own timeline, 71 events, peak −8 dBFS)

## Concept

**The photograph is the only surface, and the investigation is one camera move across it.**
The amber contour of R1 is the object that survives every boundary: it draws itself on the
photograph, is joined by R2 and R3, becomes a row in the region ledger, becomes a node in the
order graph, and its card supplies the evidence. Nothing is replaced by a new scene; the camera
travels down one continuous report column on a near-black evidence table, and the only cuts are
the end card.

Look: the product's own tokens — `#0A0B0D` ground, `#E8E5DF` paper, `#F2B544` amber for the
claim, `#FF6B6B` red for manipulation, `#5FD3A6` teal for clean, `#46CDEB`/`#EB5FD7` for the
other two regions. Type: Inter (display/text) + Geist Mono (all labels and numbers), subset and
inlined by `look.py`. No stock footage, no mock UI: the photo, the mask contours, the region
numbers, the evidence values, the order graph and the provenance findings are the demo's own.

## Beat sheet

| t (s) | beat | len | what moves | what is still | carries into |
|---|---|---|---|---|---|
| 0.00–6.30 | **hook** — a photograph, one amber contour drawing itself | 6.3 | contour draws 0.95–2.75; type lands 1.15/2.95/4.60 | 4.95–6.28 (1.3 s rest) | the contour |
| 6.30–7.65 | **pull back** — the report is revealed behind it | 1.35 | camera 2.30× → 0.80× | the board | the board / the report |
| 7.65–14.15 | **four questions** — where / order / evidence / the file | 6.5 | four rows spring in 8.15, 9.35, 10.55, 11.75 | 12.35–14.15 | the report column |
| 14.15–15.55 | **push into the photo** | 1.4 | camera 0.80× → 0.98× | — | the photograph |
| 15.55–19.85 | **three regions** — R2 and R3 draw, R1 already there | 4.3 | contours 14.85–16.35; labels 15.0–16.5 | 17.10–18.55 (1.45 s) | the region ledger |
| 19.85–23.50 | **region ledger** — three rows typed and scored | 3.65 | rows 18.65/19.07/19.49, confidence bars fill | 22.10–23.50 | R1 |
| 23.50–26.20 | **R1 close-up** — the spliced insert | 2.7 | push 1.057× → 1.95× | 25.15–26.20 | R1's node |
| 26.20–27.55 | **to the order graph** | 1.35 | camera travels down | — | the rail |
| 27.55–34.50 | **order** — R2 → R3 → R1, each arrow a measured pair | 6.95 | rail draws 26.55–27.55; nodes 27.05/28.15/29.25; meta 30.55 | 32.95–34.50 (1.55 s) | the R1 node |
| 34.50–36.20 | **arrow close-up** — p = +0.23 / −0.71 | 1.7 | push to 1.45× | 36.2 | the evidence card |
| 36.20–42.10 | **evidence** — four clues, four bars | 5.9 | rows 37.05/37.67/38.29/38.91; bars fill | 42.10–43.65 | the channel attribution |
| 42.10–43.65 | **channel attribution** — which channel carried the call | 1.55 | five bars 40.55–41.23 | — | the close-up |
| 43.65–46.20 | **+8σ close-up** | 2.55 | push 1.12× → 1.35× | 45.10–46.20 | the file |
| 46.20–51.50 | **provenance** — the gauge sweeps 0 → 72 | 5.3 | gauge 46.75–48.15; findings 48.55/49.40 | 49.70–51.50 (1.8 s) | the measurements |
| 51.50–52.75 | **pull to the measurement tape** | 1.25 | camera travels down | — | the numbers |
| 52.75–57.55 | **measured** — eight values count up | 4.8 | 8 cells 52.75–53.66; note 54.18 | 55.10–57.55 (2.45 s) | the end card |
| 57.55–58.85 | **final pull-back** — timeline → provenance as one case file | 1.3 | camera 1.37× → 0.63× | — | the whole report |
| 58.85–64.00 | **end card** — EDITTRACE, repo | 5.15 | letters drop 58.95; tagline 59.55; line 59.95; pill 60.45 | 61.20–64.00 (2.8 s) | — |

Shot lengths run 0.55 s → 6.95 s (CV 0.74). Still frames: 62 % of the cut, with eleven explicit
rests ≥ 1.2 s. One burst: the eight measurement values at 52.75–53.66.

## On-screen copy

| t | copy | sub |
|---|---|---|
| 1.15 | every detector gives you one bit | edited · not edited |
| 2.95 | edited. | or not. |
| 4.60 | that is the least interesting part | the trace is |
| 6.95 | one pass, one report | 12 channels · 18-channel U-Net · order · audit |
| 8.15–11.75 | where were the pixels edited? / in what order were the edits applied? / which evidence supports each claim? / does the file agree with itself? | 00 / FOUR QUESTIONS |
| 12.70 | one pass · four answers | — |
| 15.00 | three regions, each with a boundary | R1 splice · R2 inpaint · R3 copy-move |
| 17.00 | a mask is a claim about where the edit stops | 3 regions · 12.9% of the frame |
| 19.00 | each region is typed and scored | confidence 0.72 · 0.45 · 0.52 |
| 23.65 | R1 — a spliced insert | bbox 93,107 · 8.4% of frame |
| 27.10 | order is a precedence graph | learned pairwise model · four physical cues |
| 29.35 | cycle-safe topological sort | R2 → R3 → R1 · 3 decided pairs |
| 31.55 | each arrow is a measured pair | p = +0.23 · p = −0.71 |
| 34.60 | not a rank. a decision with evidence. | undecided pairs stay undecided |
| 37.30 | every claim carries its measurement | R1 · spliced insert · 72% confidence |
| 39.35 | and the channel that carried it | blockiness −36% · noise +24% |
| 42.10 | +8σ above the surrounding ring | recompression at a different quality |
| 46.50 | the file is evidence too | EXIF · XMP · thumbnails · quantisation |
| 48.55 | provenance 72 / 100 · suspicious | independent of the pixels |
| 50.45 | stripped metadata is a finding | no camera · no ICC profile |
| 52.55 | measured, not asserted | 12 held-out scenes · 2 edits each |
| 54.45 | ordering is 12/12 with ground-truth masks | Kendall τ +1.000 |
| 58.95 | EDITTRACE · forensic edit trace | where · in what order · on what evidence · github.com/akshaytheflash/imagetrace |

## Numbers on screen (all from the demo / README; none invented)

- 3 regions, 12.9 % edited area, peak mask 1.00, JPEG Q≈92 — `sample_01.report.html`.
- R1 splice 0.716 → 72; R2 inpaint 0.451 → 45; R3 copy-move 0.517 → 52 — `sample_01.json`.
- order R2 → R3 → R1, confidence 0.553 → 55 %, 3 decided pairs, 0 cycles — `sample_01.json`.
- R1 evidence: ELA +8.0σ, chroma +1.5σ, CFA +1.2σ, double-quant period 6; attribution
  blockiness −36 %, noise_z +24 %, ghost_min +15 %, color_shift +8 %, ghost_spread +7 %.
- provenance 0.724 → 72, no camera metadata, no ICC profile, period 3 — `sample_01.json`.
- held-out: IoU 0.516, F1 0.648, region P 0.610 / R 0.792, type 0.611, order τ 0.905,
  pairwise 0.929 (coverage 0.556), 1.10 s/image CPU; ground-truth masks → 12/12, τ 1.000.

## Assets and their sources

| asset | source |
|---|---|
| `build/assets/photo.jpg` | `.out/demo/samples/sample_01.jpg` (the demo's own image) |
| `build/assets/regions.json` | region contours traced from the mask PNG embedded in `.out/demo/sample_01.report.html`, matched to the bboxes in `sample_01.json` |
| `build/assets/regions.js` | the same paths, inlined so the comp works over `file://` |
| `launch/assets/rep_img_00.png` | the annotated report image, used only for the poster |
| report modules | rebuilt in HTML at the report's own layout and tokens (header, region ledger, order graph, evidence card, provenance card, measurement tape), driven by the real values above |
| type | Inter + Geist Mono, subset and inlined by `scripts/look.py` from `look.json` |
| sound | `build/score.py` → `build/sfx.wav`, onetake's `sfx_palette.py`; one room (T60 1.35 s), a −45 dB RMS bed and 71 placed events |

## Production notes

- 12 full-resolution forensic channels feed an 18-channel U-Net (1.1 M params); regions are typed,
  ordered by a learned pairwise precedence model + cycle-safe topological sort, then audited.
- The film deliberately shows only `sample_01`'s report: one case, followed all the way down,
  rather than a montage of three. The other two demo reports are untouched.
- The camera is keyed and operated: `OM.camera` keys with `sineInOut`/`expoInOut`/`cubicInOut`,
  a two-sine hand float that is zeroed in every rest, `OM.shake` on landings, and slow drifts on
  the long section holds.
- Render: `render.py --shutter 180 --gap 10 --samples-min 3 --samples-max 16`, 8 workers.
- No 9:16 cut: the film is authored as a 1400 px report column; a vertical reframe would crop the
  evidence card and the order graph, so it is not a cheap second export. The 16:9 master is the
  deliverable.
