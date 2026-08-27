"""Independent acceptance harness entry point.

Runs the six negative tests, evaluates the seven acceptance gates over a
baseline/candidate pair plus the data report, and writes an acceptance report
with the overall decision: ACCEPT / REVISE / HOLD.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..comparison import compare_runs
from ..data.ingest import load_manifest, validate_records
from ..data.splits import check_leakage, load_splits
from .gates import evaluate_gates
from .negative_tests import run_all_negative_tests


def build_data_report(manifest_dir: Path, splits_path: Path) -> Dict[str, Any]:
    """Data-gate evidence: licence/provenance validity + frozen acceptance."""
    manifest, records = load_manifest(manifest_dir)
    problems = validate_records(records)
    splits = load_splits(splits_path)
    leakage = check_leakage(records, splits)
    split_matches_manifest = splits.get("manifest_id") == manifest.get("manifest_id")
    records_match = (
        not splits.get("records_sha256")
        or splits.get("records_sha256") == manifest.get("records_sha256")
    )
    return {
        "valid": not problems and split_matches_manifest and records_match,
        "problems": problems[:10],
        "manifest_id": manifest["manifest_id"],
        "licences": manifest["licences"],
        "n_records": manifest["n_records"],
        "acceptance_frozen": bool(splits.get("counts", {}).get("acceptance"))
        and bool(manifest.get("split_frozen")),
        "split_unit": splits.get("split_unit"),
        "split_matches_manifest": split_matches_manifest,
        "records_match": records_match,
        "leakage_free": not leakage,
        "leakage_problems": leakage[:5],
    }


def run_acceptance(
    baseline_path: Path,
    candidate_path: Path,
    manifest_dir: Path,
    splits_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))

    data_report = build_data_report(manifest_dir, splits_path)

    neg_report = run_all_negative_tests()
    neg_report_path = Path(out_dir) / "negative_tests.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    neg_report_path.write_text(json.dumps(neg_report, indent=2), encoding="utf-8")

    # attach negative-test outcomes to the data report for the assurance gate
    tests = neg_report["tests"]
    data_report["negative_tests"] = {k: bool(v.get("pass")) for k, v in tests.items()}

    try:
        comparison = compare_runs(baseline, candidate)
    except Exception as exc:
        report = {
            "schema": "klcap-acceptance-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "overall": "HOLD",
            "reason": f"comparison blocked: {exc}",
            "data_report": data_report,
            "negative_tests": neg_report,
        }
        (Path(out_dir) / "acceptance_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    gate_report = evaluate_gates(baseline, candidate, comparison, data_report)

    report = {
        "schema": "klcap-acceptance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": baseline.get("run_id"),
        "candidate_run": candidate.get("run_id"),
        "overall": gate_report["overall"],
        "failed_gates": gate_report["failed_gates"],
        "gates": gate_report["gates"],
        "comparison": comparison,
        "data_report": data_report,
        "negative_tests": neg_report,
    }
    (Path(out_dir) / "acceptance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[acceptance] decision={report['overall']} failed={report['failed_gates']}")
    return report
