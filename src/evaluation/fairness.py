"""Subgroup-aware audit.

Computes *every* metric per approved dataset subgroup and per domain, flags
unacceptable gaps, and detects critical-slice regressions between a baseline
and a candidate — the mechanism behind negative test #4 (overall recall up,
rare slice down ⇒ REVISE).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_MAX_GAP = 0.25          # max allowed recall@1 gap between slices
CRITICAL_SUBGROUPS = {"rare-object", "lighting-low"}   # protected slices
CRITICAL_REGRESSION_TOLERANCE = 0.02   # tolerated drop on critical slices


def slice_metrics(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    gold: Sequence[int],
    subgroups: Sequence[str],
    domains: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup and per-domain retrieval metrics + disparity report."""
    from .retrieval import recall_at_k

    sims = query_embs @ corpus_embs.T
    ranks = []
    for i in range(len(gold)):
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == gold[i])[0][0]))

    def _recall(indices: Sequence[int], k: int = 1) -> float:
        if not indices:
            return float("nan")
        return float(np.mean([ranks[i] < k for i in indices]))

    all_idx = list(range(len(gold)))
    by_subgroup: Dict[str, Any] = {}
    for sg in sorted(set(subgroups)):
        idx = [i for i in all_idx if subgroups[i] == sg]
        by_subgroup[sg] = {
            "n": len(idx),
            "recall@1": _recall(idx),
            "recall@5": _recall(idx, 5),
        }
    by_domain: Dict[str, Any] = {}
    for dm in sorted(set(domains)):
        idx = [i for i in all_idx if domains[i] == dm]
        by_domain[dm] = {
            "n": len(idx),
            "recall@1": _recall(idx),
            "recall@5": _recall(idx, 5),
        }

    def _gaps(table: Dict[str, Any]) -> Dict[str, float]:
        vals = [v["recall@1"] for v in table.values() if v["n"] > 0 and not np.isnan(v["recall@1"])]
        if len(vals) < 2:
            return {}
        worst = min(vals)
        best = max(vals)
        return {
            "max_gap_recall@1": float(best - worst),
            "worst_slice_recall@1": float(worst),
            "best_slice_recall@1": float(best),
        }

    return {
        "overall": {"n": len(all_idx), "recall@1": _recall(all_idx), "recall@5": _recall(all_idx, 5)},
        "by_subgroup": by_subgroup,
        "by_domain": by_domain,
        "disparity_subgroup": _gaps(by_subgroup),
        "disparity_domain": _gaps(by_domain),
    }


def fairness_verdict(
    slices: Dict[str, Any],
    max_gap: float = DEFAULT_MAX_GAP,
) -> Dict[str, Any]:
    """Flag unacceptable subgroup gaps; never let aggregates hide a slice."""
    gap_sg = slices.get("disparity_subgroup", {}).get("max_gap_recall@1")
    gap_dm = slices.get("disparity_domain", {}).get("max_gap_recall@1")
    gaps = [g for g in (gap_sg, gap_dm) if g is not None]
    max_observed = max(gaps) if gaps else 0.0
    flagged = []
    for name, table in (("subgroup", slices.get("by_subgroup", {})),
                        ("domain", slices.get("by_domain", {}))):
        for slice_name, m in table.items():
            if m["n"] > 0 and not np.isnan(m["recall@1"]):
                if m["recall@1"] < 0.05:
                    flagged.append({
                        "slice": f"{name}:{slice_name}",
                        "reason": f"near-zero recall {m['recall@1']:.3f}",
                    })
    ok = max_observed <= max_gap and not flagged
    return {
        "fairness_ok": bool(ok),
        "max_observed_gap": float(max_observed),
        "threshold": float(max_gap),
        "flagged_slices": flagged,
    }


def critical_regression_check(
    baseline_slices: Dict[str, Any],
    candidate_slices: Dict[str, Any],
    tolerance: float = CRITICAL_REGRESSION_TOLERANCE,
) -> Dict[str, Any]:
    """Detect critical-slice regressions hidden by aggregate improvement."""
    regressions: List[Dict[str, Any]] = []
    for axis in ("by_subgroup", "by_domain"):
        btab = baseline_slices.get(axis, {})
        ctab = candidate_slices.get(axis, {})
        for slice_name, bm in btab.items():
            cm = ctab.get(slice_name)
            if cm is None or bm["n"] == 0:
                continue
            delta = cm["recall@1"] - bm["recall@1"]
            is_critical = (
                (axis == "by_subgroup" and slice_name in CRITICAL_SUBGROUPS)
                or bm["n"] < 40   # rare slice
            )
            if is_critical and delta < -tolerance:
                regressions.append({
                    "slice": f"{axis}:{slice_name}",
                    "baseline_recall@1": float(bm["recall@1"]),
                    "candidate_recall@1": float(cm["recall@1"]),
                    "delta": float(delta),
                    "critical": True,
                })
    overall_delta = (
        candidate_slices["overall"]["recall@1"] - baseline_slices["overall"]["recall@1"]
        if "overall" in baseline_slices and "overall" in candidate_slices else 0.0
    )
    return {
        "overall_delta_recall@1": float(overall_delta),
        "critical_regressions": regressions,
        "verdict": "REVISE" if regressions else "OK",
    }
