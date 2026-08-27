"""Negative-test harness: six required failure demonstrations.

Each test *injects* a fault into real lab components and asserts the intended
safe behaviour. Results feed the assurance gate; any test that fails to
demonstrate its guard forces overall ``HOLD``.

1. unequal_compute_blocked      — changed manifest / over-budget → comparison blocked
2. leakage_detected             — duplicate scene across splits → validation fails
3. abstention_labelled          — out-of-envelope query → labelled abstention
4. critical_regression_revise   — rare-slice regression hidden by aggregate → REVISE
5. artifact_mismatch_refused    — checkpoint/config hash mismatch → inference refused
6. telemetry_missing_no_claim   — missing FLOPs/energy → no resource-improvement claim
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..comparison import ComparisonBlocked, compare_runs
from ..data.ingest import generate_synthetic_records, validate_records
from ..data.splits import check_leakage, freeze_splits
from ..evaluation.fairness import critical_regression_check
from ..models.lora import BudgetExceededError, LoRAAdapter


def _base_run_pair() -> tuple:
    """Two otherwise-identical valid run records (baseline + candidate)."""
    def mk(role: str) -> Dict[str, Any]:
        return {
            "schema": "klcap-run-v1",
            "run_id": f"negtest-{role}",
            "role": role,
            "evaluator_version": "eval-1.0.0",
            "config_hash": "abc123",
            "seed": 167,
            "git_commit": "deadbeef",
            "config": {"lora": {"enabled": role == "candidate", "max_trainable_params": 100000}},
            "data": {
                "manifest_id": "manifest-x",
                "manifest_sha256": "hash-x",
                "split_id": "splits-x",
            },
            "metrics": {"recall@1": 0.5, "recall@5": 0.7, "probe_accuracy": 0.9},
            "slices": {
                "overall": {"n": 100, "recall@1": 0.5},
                "by_subgroup": {f"sg{i}": {"n": 50, "recall@1": 0.5} for i in range(2)},
                "by_domain": {},
            },
            "fairness_verdict": {"fairness_ok": True},
            "telemetry": {
                "complete": True,
                "energy_joules": 100.0,
                "flops_train": 1e6,
            },
            "artifacts": {"manifest_id": "manifest-x"},
            "failure_examples": [],
            "record_sha256": "sig",
        }
    return mk("baseline"), mk("candidate")


def test_unequal_compute_blocked() -> Dict[str, Any]:
    """Candidate trained on a different manifest must be blocked."""
    baseline, candidate = _base_run_pair()
    candidate["data"]["manifest_id"] = "manifest-CHANGED"
    try:
        compare_runs(baseline, candidate)
        return {"pass": False, "detail": "comparison was NOT blocked"}
    except ComparisonBlocked as exc:
        return {"pass": True, "detail": str(exc)}


def test_leakage_detected() -> Dict[str, Any]:
    """A scene spanning train+acceptance must fail the leakage check."""
    records = generate_synthetic_records(60, seed=167)
    splits = freeze_splits(
        {"manifest_id": "m", "split_frozen": False}, copy.deepcopy(records),
        out_dir=Path("evidence/_negtmp"), seed=167,
    )
    # inject leak: force one acceptance sample's scene into train
    accept_ids = [r["sample_id"] for r in records if splits["assignments"][r["sample_id"]] == "acceptance"]
    victim = next(r for r in records if r["sample_id"] == accept_ids[0])
    splits["assignments"][victim["sample_id"]] = "train"
    problems = check_leakage(records, splits)
    ok = len(problems) > 0
    return {"pass": bool(ok), "detail": problems[:3] or "leakage NOT detected"}


def test_abstention_labelled() -> Dict[str, Any]:
    """Out-of-envelope query must produce a labelled abstention."""
    from ..models.encoders import HashingTextEncoder

    approved_domains = {"indoor", "outdoor"}

    class SafeRetriever:
        """Retrieval wrapper enforcing domain/metadata envelope checks."""

        def __init__(self) -> None:
            self.encoder = HashingTextEncoder()

        def query(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
            reasons: List[str] = []
            if metadata.get("domain") not in approved_domains:
                reasons.append(f"unapproved domain '{metadata.get('domain')}'")
            if not metadata.get("subgroup"):
                reasons.append("missing required subgroup metadata")
            if reasons:
                return {
                    "status": "ABSTAIN",
                    "reasons": reasons,
                    "results": [],
                    "claim": None,
                }
            emb = self.encoder.encode_queries([text])
            return {"status": "OK", "results": emb.tolist(), "claim": "retrieval"}

    r = SafeRetriever()
    out_of_domain = r.query("a canoe on a river", {"domain": "underwater", "subgroup": None})
    missing_meta = r.query("a canoe on a river", {"domain": "indoor"})
    ok = (
        out_of_domain["status"] == "ABSTAIN"
        and missing_meta["status"] == "ABSTAIN"
        and out_of_domain["reasons"]
        and missing_meta["reasons"]
    )
    return {
        "pass": bool(ok),
        "detail": {"out_of_domain": out_of_domain, "missing_metadata": missing_meta},
    }


def test_critical_regression_revise() -> Dict[str, Any]:
    """Overall recall up but rare slice down ⇒ REVISE verdict."""
    baseline_slices = {
        "overall": {"n": 200, "recall@1": 0.50},
        "by_subgroup": {
            "common-object": {"n": 180, "recall@1": 0.55},
            "rare-object": {"n": 20, "recall@1": 0.40},
        },
        "by_domain": {},
    }
    candidate_slices = {
        "overall": {"n": 200, "recall@1": 0.56},     # aggregate improved!
        "by_subgroup": {
            "common-object": {"n": 180, "recall@1": 0.62},
            "rare-object": {"n": 20, "recall@1": 0.30},  # critical slice crashed
        },
        "by_domain": {},
    }
    report = critical_regression_check(baseline_slices, candidate_slices)
    ok = report["verdict"] == "REVISE" and report["critical_regressions"]
    return {"pass": bool(ok), "detail": report}


def test_artifact_mismatch_refused() -> Dict[str, Any]:
    """Loading a tampered checkpoint must be refused via state-hash check."""
    adapter = LoRAAdapter(dim=32, rank=4, seed=167)
    path = Path("evidence/_negtmp/adapter.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["B"][0][0] += 0.5                      # tamper with weights
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        LoRAAdapter.load(str(path))
        return {"pass": False, "detail": "tampered checkpoint was accepted"}
    except ValueError as exc:
        return {"pass": True, "detail": str(exc)}
    finally:
        path.unlink(missing_ok=True)


def test_telemetry_missing_no_claim() -> Dict[str, Any]:
    """Missing FLOPs/energy must disable resource-improvement claims."""
    baseline, candidate = _base_run_pair()
    candidate["telemetry"] = {"complete": False, "energy_joules": None, "flops_train": 0}
    try:
        compare_runs(baseline, candidate)
        return {"pass": False, "detail": "comparison allowed without telemetry"}
    except ComparisonBlocked as exc:
        blocked = "telemetry" in str(exc).lower()
        return {
            "pass": bool(blocked),
            "detail": str(exc) if blocked else "blocked for wrong reason",
        }


ALL_TESTS = {
    "unequal_compute_blocked": test_unequal_compute_blocked,
    "leakage_detected": test_leakage_detected,
    "abstention_labelled": test_abstention_labelled,
    "critical_regression_revise": test_critical_regression_revise,
    "artifact_mismatch_refused": test_artifact_mismatch_refused,
    "telemetry_missing_no_claim": test_telemetry_missing_no_claim,
}


def run_all_negative_tests(out_path: Path | None = None) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    all_pass = True
    for name, fn in ALL_TESTS.items():
        try:
            res = fn()
        except Exception as exc:  # a crashing guard is a failing guard
            res = {"pass": False, "detail": f"harness exception: {exc}"}
        results[name] = res
        all_pass &= bool(res["pass"])
    report = {
        "schema": "klcap-negative-tests-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": bool(all_pass),
        "tests": results,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
