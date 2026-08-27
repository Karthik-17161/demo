# Operator Guide — Running & Reviewing the Lab

KLCAP-2026-00167 · Vision-Language Representation Lab

This guide is for the **operator / faculty reviewer** who launches runs,
inspects results and exports evidence through the API and dashboard.

## 1. Start the services

### Local (no Docker)

```powershell
# API on http://localhost:8000 (interactive docs at /docs)
python -m src.cli serve

# Reviewer dashboard (needs Node.js) on http://localhost:5173
cd apps/web && npm install && npm run dev

# No Node.js? Open apps/web/standalone/index.html in any browser.
```

### Full stack via Docker Compose

```powershell
docker compose up --build
```

| Service | URL | Purpose |
|---|---|---|
| api | http://localhost:8000 | FastAPI operator service |
| dashboard | http://localhost:5173 | React reviewer dashboard |
| db | localhost:5432 | PostgreSQL experiment metadata |
| objects | http://localhost:9001 | MinIO artifact store console |

## 2. Health & quality indicators

```powershell
curl http://localhost:8000/health
```

Returns liveness plus run/evidence counters:

```json
{ "status": "ok", "n_runs": 4, "negative_tests_recorded": true, "lab_version": "1.0.0" }
```

`negative_tests_recorded: true` means the assurance evidence bundle exists.

## 3. Launch the full O2→O3 workflow

**Dashboard:** click *▶ Run full O2→O3 workflow*. It will, in order:
generate demo data → freeze splits → register + run O2 baseline →
register + run O3c candidate → fair comparison → subgroup audit →
acceptance harness → negative tests.

**API equivalent:**

```powershell
curl -X POST http://localhost:8000/datasets/demo      -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:8000/datasets/freeze    -H "Content-Type: application/json" -d "{}"

curl -X POST http://localhost:8000/experiments -H "Content-Type: application/json" ^
  -d "{\"name\":\"o2\",\"config_path\":\"configs/baseline.yaml\"}"
curl -X POST http://localhost:8000/experiments/<exp_id>/run -H "Content-Type: application/json" -d "{}"
# repeat with configs/o3c.yaml for the candidate

curl http://localhost:8000/comparisons/<baseline_run>/<candidate_run>
curl http://localhost:8000/evaluations/<candidate_run>/slices
curl -X POST http://localhost:8000/acceptance/<candidate_run>/run -H "Content-Type: application/json" ^
  -d "{\"baseline_id\":\"<baseline_run>\"}"
curl -X POST http://localhost:8000/negative-tests/run
```

## 4. Reading the results

* **Runs table** — one row per experiment with Recall@1 and probe accuracy.
* **O2 vs O3 comparison** — metric deltas; resource improvement is shown only
  when telemetry is complete (otherwise "claim disabled").
* **Subgroup audit** — per-slice recall; fairness badge shows max observed gap
  vs threshold. A red badge means a slice would be hidden by aggregates.
* **Acceptance decision** — `ACCEPT` (all gates pass), `REVISE` (a gate failed,
  e.g. critical-slice regression), or `HOLD` (assurance/negative tests missing).
  Each gate row links its pass/fail detail.
* **Negative tests** — six injected faults; each must show `DEMONSTRATED`.

## 5. Exporting evidence

Every artifact is downloadable by id:

```powershell
curl http://localhost:8000/reports/acceptance_report   # full acceptance bundle
curl http://localhost:8000/reports/<run_id>            # signed run record
curl http://localhost:8000/reports/<candidate_run>     # comparison JSON
```

On disk: `runs/*.json` (run records + comparisons),
`evidence/acceptance/acceptance_report.json`,
`evidence/negative_tests.json`, `evidence/demo/*` (manifest + records),
`evidence/splits/splits.json`.

## 6. Safe-abstention behaviour

Queries outside the approved envelope (unapproved domain or missing subgroup
metadata) return a labelled `ABSTAIN` response instead of a retrieval claim —
demonstrated live by negative test #3 (`abstention_labelled`). Operators must
not present retrieval output for out-of-envelope queries.

## 7. Recovery procedures

| Failure | Recovery |
|---|---|
| Comparison blocked (unequal data/compute) | Re-run baseline AND candidate against the same frozen manifest/splits. |
| Acceptance = REVISE (fairness) | Inspect flagged slices in `/evaluations/{id}/slices`; adjust training data balance or thresholds within approved config, re-run ablation ladder. |
| Acceptance = HOLD (assurance) | Run `POST /negative-tests/run`; all six guards must demonstrate before re-review. |
| Artifact mismatch refused | The checkpoint/config was tampered with or regenerated inconsistently — delete the stale artifact and re-run the experiment. |
| Telemetry incomplete | Verify the collector ran (see run record notes); resource-improvement claims stay disabled until complete. |

## 8. Operational boundaries

* The backend owns data/artifacts; the dashboard never touches raw records.
* The acceptance split is read-only after freezing.
* Do not raise `lora.max_trainable_params` without re-running the whole
  ablation ladder — the fair-comparison gate will block mixed-envelope pairs.
