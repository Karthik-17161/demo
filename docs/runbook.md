# Runbook — Reproducibility & Evidence Regeneration

KLCAP-2026-00167 · Vision-Language Representation Lab

This runbook rebuilds **every piece of evidence** in the lab from a clean
machine: dataset manifest, frozen splits, O2 baseline, O3 ablations,
comparison, acceptance report and negative tests. Total runtime on a laptop
CPU is a few minutes (the deterministic hashing encoder needs no GPU).

## 0. Environment

| Requirement | Value |
|---|---|
| Python | 3.11 |
| OS | Windows / Linux / macOS (pure NumPy core) |
| GPU | not required (NVML energy used automatically when present) |

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock
```

Pinned dependencies live in `requirements.lock` (numpy, fastapi, uvicorn,
pydantic, PyYAML). Optional extras (`torch`, `open_clip_torch`, `pynvml`) are
commented out; enabling them switches the encoder backend without code changes.

## 1. Freeze evidence before model work

```powershell
python -m src.cli make-demo-data --out evidence/demo --n-pairs 1500 --seed 167
python -m src.cli freeze-splits --manifest evidence/demo --out evidence/splits --seed 167
```

Produces:

* `evidence/demo/manifest.json` — manifest_id, licences (`CC-BY-4.0`),
  sources, domains, subgroups, `records_sha256`;
* `evidence/demo/records.jsonl` — one licensed pair per line with
  `sample_id`, `image_uri`, `caption`, `scene_id`, `domain`, `subgroup`,
  per-record `content_hash`;
* `evidence/splits/splits.json` — content-addressed split assignments by
  `scene_id` (the independent unit), with train/validation/acceptance counts.

The acceptance split is read-only from this point: no tuning may touch it.
Re-running `freeze-splits` with the same seed reproduces byte-identical
assignments (`split_id` is derived from the assignment content).

## 2. O2 baseline (frozen comparison point)

```powershell
python -m src.cli run --config configs/baseline.yaml
```

The signed run record lands in `runs/o2-baseline-*.json` and contains:
config hash, seed, git commit, python/platform versions, encoder identity,
manifest + split ids, Recall@1/5/10, MRR, median rank, linear-probe accuracy,
robustness drops, per-subgroup/per-domain slices, FLOPs + energy telemetry,
failure examples and an artifact registry (LoRA checkpoint SHA-256 when present).

## 3. O3 ablations (identical data + compute envelope)

```powershell
python -m src.cli run --config configs/o3a.yaml   # baseline + LoRA
python -m src.cli run --config configs/o3b.yaml   # + hard-negative curriculum
python -m src.cli run --config configs/o3c.yaml   # + subgroup-aware audit
```

All four configs share `seed: 167`, the same manifest/splits paths and the
same trainable-parameter budget (`lora.max_trainable_params: 100000`), so the
fair-comparison gate can verify envelope equality.

## 4. Compare + accept

```powershell
# fill in the two run ids printed by step 2/3
python -m src.cli compare --baseline runs/<o2-run>.json --candidate runs/<o3c-run>.json

python -m src.cli acceptance --baseline runs/<o2-run>.json --candidate runs/<o3c-run>.json `
    --manifest-dir evidence/demo --splits evidence/splits/splits.json `
    --out evidence/acceptance

python -m src.cli negative-tests --out evidence/negative_tests.json
```

Outputs:

* `runs/comparison-*.json` — fair-comparison gate verdict + metric deltas +
  resource improvement + critical-slice regression check;
* `evidence/acceptance/acceptance_report.json` — seven gates →
  `ACCEPT` / `REVISE` / `HOLD`;
* `evidence/negative_tests.json` — all six injected-fault demonstrations.

## 5. Verify reproducibility (clean-machine check)

On a second machine/container:

```powershell
docker compose build api
docker compose run --rm api python -m src.cli make-demo-data --out evidence/demo --n-pairs 1500
docker compose run --rm api python -m src.cli freeze-splits --manifest evidence/demo --out evidence/splits
docker compose run --rm api python -m src.cli run --config configs/baseline.yaml
```

Expected: identical `manifest_id`, `split_id`, `config_hash` and (modulo
wall-clock-dependent fields) identical metrics to the local run, because the
encoder, sampler and splits are all seeded deterministically.

## 6. Unit + integration tests

```powershell
pip install pytest
pytest tests/ -v
```

Covers data integrity/tamper detection, leakage resistance, LoRA budget and
checkpoint identity, curriculum logging, retrieval/probe/fairness math, the
full workflow and all six negative tests.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `splits not frozen … run freeze-splits first` | Step 1 was skipped or paths differ from config. |
| `artifact mismatch: records hash …` | `records.jsonl` edited after freezing; regenerate evidence. |
| `ComparisonBlocked: unequal data …` | Baseline/candidate trained on different manifests/splits — re-run both against the same frozen evidence. |
| `telemetry unavailable/incomplete` | Telemetry collector failed; resource claims are disabled by design until fixed. |
| Dashboard shows *API unreachable* | Start `python -m src.cli serve`; dashboard proxies `/api` to port 8000. |
