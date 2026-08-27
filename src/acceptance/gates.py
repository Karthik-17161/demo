"""Acceptance gates: the seven pass/fail conditions from the architecture.

The harness is *independent* of training code: it consumes only run records,
manifests and split files, so a gate decision cannot be influenced by the
code under test.
"""

from __future__ import annotations

from typing import Any, Dict, List

QUALITY_TARGET_RECALL1 = 0.30
NON_INFERIORITY_MARGIN = 0.05
MIN_RESOURCE_IMPROVEMENT = 0.20
MAX_FAIRNESS_GAP = 0.25


def evaluate_gates(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    comparison: Dict[str, Any],
    data_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a full gate report with an overall ACCEPT / REVISE / HOLD."""
    gates: List[Dict[str, Any]] = []

    # 1. Data gate
    gates.append({
        "gate": "data",
        "pass": bool(data_report.get("valid"))
        and bool(data_report.get("acceptance_frozen")),
        "detail": data_report,
    })

    # 2. Reproducibility gate
    repro_fields = ("config_hash", "seed", "git_commit")
    missing = [
        f"{side}.{f}"
        for side, run in (("baseline", baseline), ("candidate", candidate))
        for f in repro_fields if not run.get(f)
    ]
    gates.append({
        "gate": "reproducibility",
        "pass": not missing,
        "detail": {"missing": missing},
    })

    # 3. Fair-comparison gate (comparison would have raised otherwise)
    gates.append({
        "gate": "fair_comparison",
        "pass": comparison.get("gate") == "PASS",
        "detail": {
            "manifest_baseline": baseline.get("data", {}).get("manifest_id"),
            "manifest_candidate": candidate.get("data", {}).get("manifest_id"),
        },
    })

    # 4. Quality gate (target OR non-inferiority + ≥20% resource improvement)
    delta_r1 = comparison["metrics_delta"]["recall@1"]["delta"] or 0.0
    cand_r1 = comparison["metrics_delta"]["recall@1"]["candidate"] or 0.0
    base_r1 = comparison["metrics_delta"]["recall@1"]["baseline"] or 0.0
    res = comparison.get("resource", {}).get("resource_improvement_fraction")
    quality_pass = (
        cand_r1 >= QUALITY_TARGET_RECALL1
        or (delta_r1 >= -NON_INFERIORITY_MARGIN and (res or 0) >= MIN_RESOURCE_IMPROVEMENT)
    )
    gates.append({
        "gate": "quality",
        "pass": bool(quality_pass),
        "detail": {
            "candidate_recall@1": cand_r1,
            "baseline_recall@1": base_r1,
            "delta": delta_r1,
            "resource_improvement": res,
            "required_resource_improvement": MIN_RESOURCE_IMPROVEMENT,
        },
    })

    # 5. Fairness gate — no critical subgroup hidden by aggregates
    fv = candidate.get("fairness_verdict", {})
    regression = comparison.get("critical_regression_check", {})
    fairness_pass = bool(fv.get("fairness_ok")) and regression.get("verdict") == "OK"
    gates.append({
        "gate": "fairness",
        "pass": bool(fairness_pass),
        "detail": {
            "candidate_fairness_verdict": fv,
            "critical_regressions": regression.get("critical_regressions", []),
        },
    })

    # 6. Operations gate — logs/artifacts/telemetry/exportable evidence exist
    ops_ok = all([
        bool(candidate.get("telemetry", {}).get("complete")),
        bool(candidate.get("artifacts", {}).get("manifest_id")),
        bool(candidate.get("failure_examples") is not None),
        bool(candidate.get("record_sha256")),
    ])
    gates.append({"gate": "operations", "pass": bool(ops_ok), "detail": {}})

    # 7. Assurance gate — negative tests must have demonstrated HOLD behaviour
    assurance = data_report.get("negative_tests", {})
    expected = {
        "unequal_compute_blocked", "leakage_detected", "abstention_labelled",
        "critical_regression_revise", "artifact_mismatch_refused",
        "telemetry_missing_no_claim",
    }
    seen = set(k for k, v in assurance.items() if v)
    gates.append({
        "gate": "assurance",
        "pass": expected.issubset(seen),
        "detail": {"demonstrated": sorted(seen), "expected": sorted(expected)},
    })

    failed = [g["gate"] for g in gates if not g["pass"]]
    if any(g["gate"] == "assurance" and not g["pass"] for g in gates):
        overall = "HOLD"
    elif failed:
        overall = "REVISE"
    else:
        overall = "ACCEPT"
    return {"overall": overall, "gates": gates, "failed_gates": failed}
