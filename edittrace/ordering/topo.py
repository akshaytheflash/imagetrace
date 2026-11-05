"""Cycle-safe topological ordering of edit events."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from ..config import Config
from .cues import PairEvidence


@dataclass
class OrderResult:
    sequence: list[int] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    undecided: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    cycles_broken: int = 0
    decided_pairs: int = 0

    def rank(self, region_id: int) -> int:
        return self.sequence.index(region_id) + 1 if region_id in self.sequence else -1


def _reachable(adj: dict[int, set[int]], src: int, dst: int) -> bool:
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


def order_from_evidence(pairs: list[PairEvidence], cfg: Config,
                        region_conf: dict[int, float] | None = None) -> OrderResult:
    """Build a DAG from pairwise scores, breaking conflicts by confidence.

    Edges are inserted strongest-first and skipped if they would close a
    cycle, so a single contradictory cue cannot corrupt the sequence. Skipped
    edges are reported instead of hidden.
    """
    res = OrderResult()
    region_conf = region_conf or {}
    eps = cfg.precedence_eps

    directed: list[tuple[float, int, int, PairEvidence]] = []
    for ev in pairs:
        if abs(ev.score) <= eps:
            res.undecided.append({
                "a": ev.a, "b": ev.b, "score": ev.score, "cues": ev.cues,
                "reason": "below decision threshold",
            })
            continue
        if ev.score > 0:
            directed.append((ev.score, ev.a, ev.b, ev))
        else:
            directed.append((-ev.score, ev.b, ev.a, ev))
    directed.sort(key=lambda t: t[0], reverse=True)

    adj: dict[int, set[int]] = {}
    indeg: dict[int, int] = {}
    nodes = {p.a for p in pairs} | {p.b for p in pairs}
    for n in nodes:
        adj[n] = set()
        indeg[n] = 0

    for strength, u, v, ev in directed:
        if _reachable(adj, v, u):
            res.cycles_broken += 1
            res.undecided.append({
                "a": ev.a, "b": ev.b, "score": ev.score, "cues": ev.cues,
                "reason": "contradicts stronger evidence (cycle avoided)",
            })
            continue
        if v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1
            res.edges.append({"before": u, "after": v, "score": ev.score,
                              "cues": ev.cues, "strength": strength})
    res.decided_pairs = len(res.edges)

    # Kahn's algorithm; ties prefer the more confident region.
    heap = [(-region_conf.get(n, 0.5), n) for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    seq: list[int] = []
    while heap:
        _, n = heapq.heappop(heap)
        seq.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, (-region_conf.get(m, 0.5), m))
    # Any residual nodes mean a cycle slipped through numeric corner cases.
    leftover = [n for n in nodes if n not in seq]
    if leftover:
        seq.extend(sorted(leftover))
        res.cycles_broken += len(leftover) - 1

    res.sequence = seq
    if res.edges:
        mean_strength = sum(abs(e["score"]) for e in res.edges) / len(res.edges)
        coverage = len(res.edges) / max(1, len(nodes) * (len(nodes) - 1) / 2)
        res.confidence = float(min(0.96, mean_strength * (0.55 + 0.45 * coverage)))
    return res


def kendall_tau(gt: list[int], pred: list[int]) -> float:
    """Rank correlation over the regions present in both sequences."""
    common = [x for x in gt if x in pred]
    if len(common) < 2:
        return float("nan")
    pos_pred = {v: i for i, v in enumerate(pred)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            concordant += 1 if pos_pred[a] < pos_pred[b] else 0
            discordant += 1 if pos_pred[a] > pos_pred[b] else 0
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")
