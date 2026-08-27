"""FastAPI operator service — the Operation + Integration boxes.

The backend owns access to data and artifacts; the dashboard only presents
results. State is persisted as JSON records under ``state/`` (PostgreSQL in
Docker via ``DATABASE_URL``; the JSON store keeps local dev dependency-free).

Endpoints (architecture contract):
    POST /datasets/validate            quality checks on a manifest dir
    POST /datasets/freeze              freeze leakage-resistant splits
    POST /experiments                  register an experiment from a config
    POST /experiments/{id}/run         execute it (synchronously in-process)
    GET  /comparisons/{base}/{cand}    fair-comparison gate + deltas
    GET  /evaluations/{id}/slices      subgroup audit for one run
    POST /acceptance/{cand}/run        independent acceptance harness
    POST /negative-tests/run           six failure demonstrations
    GET  /reports/{id}                 export evidence bundle
    GET  /health                       liveness/quality indicators
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import hashlib

from src.acceptance.harness import build_data_report, run_acceptance
from src.acceptance.negative_tests import run_all_negative_tests
from src.comparison import ComparisonBlocked, compare_runs, load_run
from src.data.ingest import build_manifest, generate_synthetic_records, load_manifest, validate_records
from src.data.splits import freeze_splits
from src.pipeline import load_config, run_experiment

STATE_DIR = Path("state")
RUNS_DIR = Path("runs")
EVIDENCE_DIR = Path("evidence")

app = FastAPI(
    title="KLCAP-2026-00167 Vision-Language Lab",
    version="1.0.0",
    description="Operator API: launch approved runs and review/export evidence.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # reviewer dashboard origin(s) in deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()


# ----------------------------- models -----------------------------

class DatasetRequest(BaseModel):
    manifest_dir: str = Field(..., description="Directory containing manifest.json")


class FreezeRequest(BaseModel):
    manifest_dir: str = "evidence/demo"
    out_dir: str = "evidence/splits"
    seed: int = 167


class DemoDataRequest(BaseModel):
    out_dir: str = "evidence/demo"
    n_pairs: int = 1500
    seed: int = 167


class ExperimentCreate(BaseModel):
    name: str
    config_path: str


class RunRequest(BaseModel):
    runs_dir: str = "runs"


class AcceptanceRequest(BaseModel):
    baseline_id: str
    manifest_dir: str = "evidence/demo"
    splits_path: str = "evidence/splits/splits.json"


# ----------------------------- state -----------------------------

def _load_state(name: str) -> Dict[str, Any]:
    path = STATE_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(name: str, data: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _find_run(run_id: str) -> Optional[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    exact = RUNS_DIR / f"{run_id}.json"
    if exact.exists():
        return exact
    matches = sorted(RUNS_DIR.glob(f"{run_id}*.json"))
    return matches[0] if matches else None


def _resolve_config_path(config_path: str) -> Path:
    p = Path(config_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        # allow referencing configs by experiment name registered earlier
        exps = _load_state("experiments")
        rec = exps.get(config_path)
        if rec:
            p = Path(rec["config_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"config not found: {config_path}")
    return p


# ----------------------------- endpoints -----------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    runs = list(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []
    neg = EVIDENCE_DIR / "negative_tests.json"
    return {
        "status": "ok",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "negative_tests_recorded": neg.exists(),
        "lab_version": "1.0.0",
    }


@app.post("/datasets/demo")
def make_demo(req: DemoDataRequest) -> Dict[str, Any]:
    records = generate_synthetic_records(req.n_pairs, seed=req.seed)
    manifest = build_manifest(records, Path(req.out_dir), name=f"demo{req.n_pairs}")
    return {"manifest_id": manifest["manifest_id"], "n_records": manifest["n_records"],
            "licences": manifest["licences"]}


@app.post("/datasets/validate")
def datasets_validate(req: DatasetRequest) -> Dict[str, Any]:
    try:
        manifest, records = load_manifest(Path(req.manifest_dir))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"manifest load failed: {exc}")
    problems = validate_records(records)
    return {
        "valid": not problems,
        "manifest_id": manifest["manifest_id"],
        "n_records": manifest["n_records"],
        "licences": manifest["licences"],
        "problems": problems[:20],
    }


@app.post("/datasets/freeze")
def datasets_freeze(req: FreezeRequest) -> Dict[str, Any]:
    try:
        manifest, records = load_manifest(Path(req.manifest_dir))
        splits = freeze_splits(
            manifest, records, out_dir=Path(req.out_dir), seed=req.seed,
            manifest_dir=Path(req.manifest_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"freeze failed: {exc}")
    return {"split_id": splits["split_id"], "counts": splits["counts"],
            "scene_counts": splits["scene_counts"], "split_unit": splits["split_unit"]}


@app.post("/experiments")
def create_experiment(req: ExperimentCreate) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(req.config_path)
    cfg = load_config(cfg_path)
    exp_id = f"{req.name}-{uuid.uuid4().hex[:8]}"
    with _lock:
        exps = _load_state("experiments")
        exps[exp_id] = {
            "experiment_id": exp_id,
            "name": req.name,
            "config_path": str(cfg_path),
            "config_hash": hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()[:16],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runs": [],
        }
        _save_state("experiments", exps)
    return {"experiment_id": exp_id, "config_path": str(cfg_path)}


@app.post("/experiments/{experiment_id}/run")
def run_experiments(experiment_id: str, req: RunRequest) -> Dict[str, Any]:
    exps = _load_state("experiments")
    exp = exps.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"unknown experiment {experiment_id}")
    try:
        record = run_experiment(Path(exp["config_path"]), Path(req.runs_dir))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"run failed: {exc}")
    with _lock:
        exp["runs"].append(record["run_id"])
        _save_state("experiments", exps)
    return {
        "run_id": record["run_id"],
        "recall@1": record["metrics"].get("recall@1"),
        "probe_accuracy": record["metrics"].get("probe_accuracy"),
        "fairness_ok": record["fairness_verdict"]["fairness_ok"],
        "record_path": str(RUNS_DIR / f"{record['run_id']}.json"),
    }


@app.get("/comparisons/{baseline_id}/{candidate_id}")
def comparisons(baseline_id: str, candidate_id: str) -> Dict[str, Any]:
    b_path, c_path = _find_run(baseline_id), _find_run(candidate_id)
    if not b_path or not c_path:
        raise HTTPException(status_code=404, detail="run record(s) not found")
    try:
        report = compare_runs(load_run(b_path), load_run(c_path))
    except ComparisonBlocked as exc:
        raise HTTPException(status_code=409, detail=f"comparison blocked: {exc}")
    out = RUNS_DIR / f"comparison-{candidate_id}-vs-{baseline_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@app.get("/evaluations/{run_id}/slices")
def evaluation_slices(run_id: str) -> Dict[str, Any]:
    r_path = _find_run(run_id)
    if not r_path:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    record = load_run(r_path)
    return {
        "run_id": record["run_id"],
        "overall": record["slices"].get("overall"),
        "by_subgroup": record["slices"].get("by_subgroup"),
        "by_domain": record["slices"].get("by_domain"),
        "disparity_subgroup": record["slices"].get("disparity_subgroup"),
        "disparity_domain": record["slices"].get("disparity_domain"),
        "fairness_verdict": record.get("fairness_verdict"),
        "failure_examples": record.get("failure_examples", []),
    }


@app.post("/acceptance/{candidate_id}/run")
def acceptance(candidate_id: str, req: AcceptanceRequest) -> Dict[str, Any]:
    b_path, c_path = _find_run(req.baseline_id), _find_run(candidate_id)
    if not b_path or not c_path:
        raise HTTPException(status_code=404, detail="run record(s) not found")
    report = run_acceptance(
        baseline_path=b_path,
        candidate_path=c_path,
        manifest_dir=Path(req.manifest_dir),
        splits_path=Path(req.splits_path),
        out_dir=EVIDENCE_DIR / "acceptance",
    )
    return report


@app.post("/negative-tests/run")
def negative_tests() -> Dict[str, Any]:
    report = run_all_negative_tests(out_path=EVIDENCE_DIR / "negative_tests.json")
    return report


@app.get("/reports/{report_id}")
def reports(report_id: str) -> Dict[str, Any]:
    """Export an evidence bundle by id: acceptance-*, comparison-*, or a run id."""
    candidates = [
        EVIDENCE_DIR / "acceptance" / f"{report_id}.json",
        EVIDENCE_DIR / "acceptance" / "acceptance_report.json",
        EVIDENCE_DIR / f"{report_id}.json",
        RUNS_DIR / f"{report_id}.json",
        RUNS_DIR / f"comparison-{report_id}.json",
    ]
    if report_id.startswith("acceptance"):
        candidates.insert(0, EVIDENCE_DIR / "acceptance" / "acceptance_report.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"no report found for {report_id}")


@app.get("/experiments")
def list_experiments() -> Dict[str, Any]:
    return {"experiments": list(_load_state("experiments").values())}


@app.get("/runs")
def list_runs() -> Dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        if p.name.startswith("comparison"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        runs.append({
            "run_id": r["run_id"],
            "experiment_name": r.get("experiment_name"),
            "role": r.get("role"),
            "recall@1": r["metrics"].get("recall@1"),
            "probe_accuracy": r["metrics"].get("probe_accuracy"),
            "created_utc": r.get("created_utc"),
        })
    return {"runs": runs}
