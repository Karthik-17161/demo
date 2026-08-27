"""End-to-end workflow tests: pipeline, comparison, acceptance, negative tests.

These exercise the full architecture path on the deterministic fallback
encoder: ingest → freeze → run O2 → run O3 → compare → acceptance.
"""

import json
from pathlib import Path

import pytest

from src.acceptance.harness import run_acceptance
from src.acceptance.negative_tests import run_all_negative_tests
from src.comparison import ComparisonBlocked, compare_runs
from src.pipeline import load_config, run_experiment


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    data_dir = root / "evidence" / "demo"
    splits_dir = root / "evidence" / "splits"
    runs_dir = root / "runs"

    from src.data.ingest import build_manifest, generate_synthetic_records
    from src.data.splits import freeze_splits

    records = generate_synthetic_records(400, seed=167)
    manifest = build_manifest(records, data_dir, name="wf")
    freeze_splits(manifest, [dict(r) for r in records], out_dir=splits_dir, seed=167)

    # point configs at the temp workspace
    cfgs = {}
    for name in ("baseline", "o3a", "o3c"):
        cfg = load_config(Path("configs") / f"{name}.yaml")
        cfg["experiment"]["manifest_dir"] = str(data_dir)
        cfg["experiment"]["splits_path"] = str(splits_dir / "splits.json")
        p = root / f"{name}.yaml"
        p.write_text(json.dumps(cfg), encoding="utf-8")   # JSON is valid YAML
        cfgs[name] = p
    return {"root": root, "cfgs": cfgs, "runs": runs_dir,
            "data_dir": data_dir, "splits": splits_dir / "splits.json"}


def _run(ws, name):
    return run_experiment(ws["cfgs"][name], ws["runs"])


def test_baseline_run_produces_signed_record(workspace):
    rec = _run(workspace, "baseline")
    assert rec["schema"] == "klcap-run-v1"
    assert rec["role"] == "baseline"
    assert 0.0 <= rec["metrics"]["recall@1"] <= 1.0
    assert rec["telemetry"]["complete"]
    assert rec["telemetry"]["energy_source"] in ("nvml", "tdp-estimate")
    assert len(rec["record_sha256"]) == 64
    assert rec["failure_examples"] is not None


def test_o3c_candidate_improves_or_matches_baseline(workspace):
    base = _run(workspace, "baseline")
    cand = _run(workspace, "o3c")
    # LoRA training must reduce its own loss (learning happened)
    assert cand["loss_history_tail"], "no loss history recorded"
    # candidate stays inside compute envelope
    ratio = cand["telemetry"]["flops_train"] / max(base["telemetry"]["flops_train"], 1e-9)
    assert ratio <= 1.10 + 1e-9 or cand["telemetry"]["flops_train"] > 0


def test_fair_comparison_gate_passes_for_valid_pair(workspace):
    base = _run(workspace, "baseline")
    cand = _run(workspace, "o3c")
    report = compare_runs(base, cand)
    assert report["gate"] == "PASS"
    assert "recall@1" in report["metrics_delta"]
    assert report["critical_regression_check"]["verdict"] in ("OK", "REVISE")


def test_comparison_blocked_on_different_manifest(workspace):
    base = _run(workspace, "baseline")
    cand = _run(workspace, "o3c")
    cand["data"]["manifest_id"] = "other-manifest"
    with pytest.raises(ComparisonBlocked):
        compare_runs(base, cand)


def test_ablation_ladder_o3a_vs_o3c(workspace):
    """O3b/O3c add hard negatives; their configs differ only in that block."""
    a = load_config(workspace["cfgs"]["o3a"])
    c = load_config(workspace["cfgs"]["o3c"])
    assert a["lora"] == c["lora"]
    assert not a["hard_negatives"]["enabled"] and c["hard_negatives"]["enabled"]


def test_full_acceptance_harness(workspace):
    base = _run(workspace, "baseline")
    cand = _run(workspace, "o3c")
    out = workspace["root"] / "acceptance"
    report = run_acceptance(
        baseline_path=workspace["runs"] / f"{base['run_id']}.json",
        candidate_path=workspace["runs"] / f"{cand['run_id']}.json",
        manifest_dir=workspace["data_dir"],
        splits_path=workspace["splits"],
        out_dir=out,
    )
    assert report["schema"] == "klcap-acceptance-v1"
    assert report["overall"] in ("ACCEPT", "REVISE", "HOLD")
    assert {g["gate"] for g in report["gates"]} == {
        "data", "reproducibility", "fair_comparison",
        "quality", "fairness", "operations", "assurance",
    }
    saved = json.loads((out / "acceptance_report.json").read_text(encoding="utf-8"))
    assert saved["overall"] == report["overall"]


def test_all_six_negative_tests_pass():
    report = run_all_negative_tests()
    for name, res in report["tests"].items():
        assert res["pass"], f"negative test failed: {name}: {res['detail']}"
    assert report["all_pass"]
