"""Per-region evidence: measured cues in, human-readable claims out."""

from __future__ import annotations

import numpy as np

from ..forensics.score import CLASSICAL_WEIGHTS, classical_score
from ..forensics.stack import FeatureStack
from ..regions.extract import Region


def attribute_channels(stack: FeatureStack, regions: list[Region]) -> None:
    """Ablate each classical channel and measure its contribution per region.

    This is occlusion attribution on the scoring function itself: replace one
    channel with its image mean, recompute, and look at the drop inside the
    region. Cheap, model-agnostic, and directly interpretable.
    """
    base = classical_score(stack)
    original = {k: v.copy() for k, v in stack.maps.items()}
    for name, w in CLASSICAL_WEIGHTS.items():
        if w == 0.0:
            continue
        stack.maps[name] = np.full_like(original[name], float(original[name].mean()))
        ablated = classical_score(stack)
        stack.maps[name] = original[name]
        delta = base - ablated
        for r in regions:
            r.attribution[name] = float(delta[r.mask > 0].mean())
    # normalise to shares within each region for display
    for r in regions:
        total = sum(abs(v) for v in r.attribution.values()) + 1e-9
        r.attribution = {k: v / total for k, v in r.attribution.items()}


def _strength(x: float, scale: float) -> float:
    return float(np.clip(abs(x) / scale, 0.0, 1.0))


def build_evidence(region: Region, stack: FeatureStack) -> list[dict]:
    st = region.stats
    out: list[dict] = []

    def add(clue, reading, strength, value):
        out.append({"clue": clue, "reading": reading,
                    "strength": round(float(np.clip(strength, 0.02, 1.0)), 3),
                    "value": value})

    noise_eff = st["noise_eff"]
    if abs(noise_eff) >= 0.9:
        direction = "higher" if noise_eff > 0 else "lower"
        add("Sensor-noise floor",
            f"Local noise is {direction} inside the region "
            f"({st['noise_in']:.2f} vs {st['noise_ring']:.2f} in the surrounding ring, "
            f"{noise_eff:+.1f}σ). "
            + ("Foreign grain or a different source sensor." if noise_eff > 0
               else "Smoothed or synthesized content - typical of fills and re-renders."),
            _strength(noise_eff, 3.0), f"{noise_eff:+.1f}σ")

    if abs(st["ela_eff"]) >= 0.8:
        add("Compression residual (ELA)",
            f"Recompression error is {st['ela_eff']:+.1f}σ against the ring, so this "
            "area converged differently at the estimated quality. That happens when "
            "pixels were written by a different compression history.",
            _strength(st["ela_eff"], 2.5), f"{st['ela_eff']:+.1f}σ")

    if st["ghost_q"] > 0 and abs(st["ghost_q"] - st["ghost_q_ring"]) >= 10:
        add("JPEG ghost sweep",
            f"The region is best explained at Q≈{int(st['ghost_q'])} while its "
            f"surroundings minimise at Q≈{int(st['ghost_q_ring'])}. Disagreeing ghost "
            "minima mean the two areas were last quantised separately.",
            0.7, f"Q{int(st['ghost_q'])} vs Q{int(st['ghost_q_ring'])}")

    if st["dq_score"] >= 2.5 and st["dq_q"] >= 4:
        weak = " (weak)" if st["dq_score"] < 3.2 else ""
        add(f"Double-quantisation period{weak}",
            f"Coefficient histogram has a period of {int(st['dq_q'])} "
            f"(score {st['dq_score']:.1f}), a possible fingerprint of an earlier save at "
            f"Q≈{int(st['dq_q'])}. "
            + ("This region carries the host's earlier history." if st["dq_q"] == st["dq_global_q"]
               else "The period differs from the host, suggesting a separate lineage."),
            _strength(st["dq_score"] - 1.0, 3.0), f"period {int(st['dq_q'])}")

    if st["resample"] >= 0.22:
        add("Resampling periodicity",
            f"Second-derivative autocorrelation peaks at {st['resample']:.2f}. "
            "Interpolated lattices leave this periodicity, so the patch was probably "
            "scaled or rotated after it was created.",
            _strength(st["resample"] - 0.15, 0.45), f"{st['resample']:.2f}")

    if abs(st["cfa_eff"]) >= 1.0:
        add("CFA / demosaic consistency",
            f"2x2 colour-filter residual is {st['cfa_eff']:+.1f}σ from the ring. "
            "The region does not share the host's demosaicing fingerprint, which is "
            "expected for rendered, synthetic or heavily resampled inserts.",
            _strength(st["cfa_eff"], 2.5), f"{st['cfa_eff']:+.1f}σ")

    if st.get("clone_ncc", 0) >= 0.88:
        src = region.copy_source
        where = f" at ({src[0]},{src[1]})" if src else ""
        add("Cloned texture",
            f"Gradient-domain template match reaches NCC {st['clone_ncc']:.3f}{where}. "
            "The texture exists twice in the frame, which is direct copy-move evidence.",
            _strength(st["clone_ncc"] - 0.85, 0.15), f"NCC {st['clone_ncc']:.3f}")

    if st["seam_hot"] >= 0.18:
        add("Manipulation boundary",
            f"{st['seam_hot'] * 100:.0f}% of the region outline sits on a texture edge "
            "that coincides with a statistical discontinuity - the geometric footprint "
            "of a compositing seam.",
            _strength(st["seam_hot"], 0.55), f"{st['seam_hot'] * 100:.0f}% of outline")

    if abs(st["block_eff"]) >= 1.0:
        add("Block-grid mismatch",
            "8x8 energy on the region's boundary grid is "
            f"{st['block_eff']:+.1f}σ from the ring, indicating a paste that was not "
            "aligned to the host's JPEG lattice.",
            _strength(st["block_eff"], 2.5), f"{st['block_eff']:+.1f}σ")

    if abs(st["color_eff"]) >= 1.2:
        add("Chromatic discontinuity",
            f"Local chroma deviates {st['color_eff']:+.1f}σ from the ring, a common "
            "side effect of colour-matching a pasted patch.",
            _strength(st["color_eff"], 3.0), f"{st['color_eff']:+.1f}σ")

    out.sort(key=lambda e: e["strength"], reverse=True)
    if not out:
        out.append({
            "clue": "Fused mask evidence only",
            "reading": "No single forensic cue exceeded its reporting threshold. "
                       "The region is retained on the fused mask alone, so treat its "
                       "type and rank as provisional.",
            "strength": round(float(np.clip(region.mean_prob * 0.45, 0.05, 0.4)), 3),
            "value": f"p={region.mean_prob:.2f}",
        })
    region.evidence = out[:5]
    return region.evidence
