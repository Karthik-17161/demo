"""End-to-end experiment pipeline: the O2/O3 comparison workflow.

One entry point :func:`run_experiment` executes the full architecture path:

    approved manifest → frozen splits → pinned encoder
    → (optional LoRA + hard-negative curriculum)
    → retrieval / probe / robustness / subgroup-slice metrics
    → FLOPs + energy telemetry
    → signed run record (JSON) with failure examples

The run record is the atomic unit of evidence consumed by comparisons,
the acceptance harness and the API/reports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import EVALUATOR_VERSION, __version__
from .data.ingest import load_manifest, validate_records
from .data.splits import check_leakage, load_splits
from .evaluation.fairness import fairness_verdict, slice_metrics
from .evaluation.retrieval import (
    linear_probe_accuracy,
    recall_at_k,
    robustness_perturb,
)
from .models.encoders import build_encoder
from .models.hard_negatives import curriculum_schedule, sample_curriculum
from .models.lora import BudgetExceededError, LoRAAdapter
from .telemetry.collector import energy_per_query, estimate_flops, finish_run, start_run


class PipelineError(RuntimeError):
    """Raised when a run violates a pre-condition (budget, leakage, integrity)."""


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config; uses PyYAML if present, else a tiny safe subset parser."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            new: Dict[str, Any] = {}
            parent[key] = new
            stack.append((indent, new))
        else:
            parent[key] = _coerce(value)
    return root


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.startswith("[") and v.endswith("]"):
        items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        return [_coerce(x) for x in items]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip("'\"")


def run_experiment(config_path: Path, runs_dir: Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    run_id = f"{exp['name']}-{int(time.time())}-{_sha256_obj(cfg)[:8]}"
    tel = start_run(run_id)

    # ---------- 1. evidence gate: manifest + quality checks ----------
    manifest_dir = Path(exp["manifest_dir"])
    manifest, records = load_manifest(manifest_dir)   # verifies records hash
    problems = validate_records(records)
    if problems:
        raise PipelineError(f"data gate failed: {problems[:5]}")

    splits_path = Path(
        exp.get("splits_path", str(manifest_dir.parent / "splits" / "splits.json"))
    )
    if not splits_path.exists():
        raise PipelineError(
            f"splits not frozen at {splits_path}; run freeze-splits before any model work"
        )
    splits = load_splits(splits_path)
    if splits.get("manifest_id") != manifest.get("manifest_id"):
        raise PipelineError("split manifest does not match the approved evidence manifest")
    if splits.get("records_sha256") and splits["records_sha256"] != manifest.get("records_sha256"):
        raise PipelineError("split evidence hash does not match the approved manifest")
    leakage = check_leakage(records, splits)
    if leakage:
        raise PipelineError(f"split leakage detected: {leakage[:5]}")

    for r in records:
        r["split"] = splits["assignments"][r["sample_id"]]

    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    accept = [r for r in records if r["split"] == "acceptance"]
    if not accept:
        raise PipelineError("acceptance split is empty")

    # ---------- 2. pinned encoder ----------
    encoders = build_encoder(cfg)
    txt_enc, img_enc = encoders["text"], encoders["image"]
    seed = int(cfg.get("seed", 167))

    def encode_texts_of(recs):
        return txt_enc.encode_texts([r["caption"] for r in recs]).astype(np.float32)

    def encode_split(split_records):
        texts = [r["caption"] for r in split_records]
        uris = [r["image_uri"] for r in split_records]
        scenes = [r["scene_id"] for r in split_records]
        t = txt_enc.encode_texts(texts)
        i = img_enc.encode_images(uris, scenes)
        return t.astype(np.float32), i.astype(np.float32)

    tr_txt, tr_img = encode_split(train)

    # ---------- 3. contribution: LoRA + hard-negative curriculum ----------
    lora_cfg = cfg.get("lora", {}) or {}
    use_lora = bool(lora_cfg.get("enabled", False))
    adapter: Optional[LoRAAdapter] = None
    negative_logs: List[Dict[str, Any]] = []
    loss_history: List[float] = []
    schedule: List[str] = []

    if use_lora:
        dim = tr_txt.shape[1]
        adapter = LoRAAdapter(
            dim=dim,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 8.0)),
            seed=seed,
            gated=bool(lora_cfg.get("gated", True)),
        )
        budget = int(lora_cfg.get("max_trainable_params", 100_000))
        adapter.check_budget(budget)   # raises BudgetExceededError if over

        rounds = int(lora_cfg.get("rounds", 3))
        epochs = int(lora_cfg.get("epochs", 30))
        hn_cfg = cfg.get("hard_negatives", {}) or {}
        use_hn = bool(hn_cfg.get("enabled", False))
        if use_hn:
            schedule = curriculum_schedule(rounds)
        for rnd in range(rounds):
            stage = schedule[rnd] if schedule else None
            hard_pairs = None
            if use_hn and stage:
                cur = sample_curriculum(
                    train,
                    stage=stage,
                    seed=seed + rnd,
                    max_pairs=int(hn_cfg.get("max_pairs", 256)),
                )
                hard_pairs = cur.pair_indices
                negative_logs.extend(cur.negative_log)
            hist = adapter.fit(
                tr_img, tr_txt,
                hard_pair_indices=hard_pairs,
                epochs=epochs,
                lr=float(lora_cfg.get("lr", 0.05)),
            )
            loss_history.extend(hist)

    def apply_img(e: np.ndarray) -> np.ndarray:
        return adapter.apply(e) if adapter is not None else e

    def apply_txt(e: np.ndarray) -> np.ndarray:
        return adapter.apply(e) if adapter is not None else e

    # ---------- 4. evaluation on frozen acceptance split ----------
    ac_txt, ac_img = encode_split(accept)
    ac_txt_a = apply_txt(ac_txt)
    ac_img_a = apply_img(ac_img)
    gold = list(range(len(accept)))

    metrics: Dict[str, Any] = {}
    metrics.update(recall_at_k(ac_txt_a, ac_img_a, gold, ks=(1, 5, 10)))
    metrics.update(robustness_perturb(
        [r["caption"] for r in accept],
        lambda ts: apply_txt(txt_enc.encode_texts(ts).astype(np.float32)),
        ac_img_a, gold,
    ))

    # linear probe: predict domain from image embeddings
    domains_sorted = sorted({r["domain"] for r in records})
    labels_tr = [domains_sorted.index(r["domain"]) for r in train]
    labels_ac = [domains_sorted.index(r["domain"]) for r in accept]
    tr_img_a = apply_img(tr_img)
    probe = linear_probe_accuracy(tr_img_a, labels_tr, ac_img_a, labels_ac)
    metrics["probe_accuracy"] = probe["probe_accuracy"]
    metrics["probe_per_class"] = probe["probe_per_class"]

    slices = slice_metrics(
        ac_txt_a, ac_img_a, gold,
        subgroups=[r["subgroup"] for r in accept],
        domains=[r["domain"] for r in accept],
    )
    fv = fairness_verdict(slices, max_gap=float((cfg.get("fairness", {}) or {}).get("max_gap", 0.25)))

    # ---------- 5. telemetry ----------
    total_epochs = int(lora_cfg.get("epochs", 30)) * int(lora_cfg.get("rounds", 3)) if use_lora else 0
    flops = estimate_flops(
        n_pairs=len(train),
        dim=tr_txt.shape[1],
        epochs=total_epochs,
        lora_rank=int(lora_cfg.get("rank", 8)) if use_lora else 0,
        hard_negatives=len(negative_logs),
    )
    tel = finish_run(tel, flops_train=flops["flops_train"], flops_eval=flops["flops_eval"])
    epq = energy_per_query(tel, len(accept))

    # ---------- 6. failure examples (not only averages) ----------
    sims = ac_txt_a @ ac_img_a.T
    worst = []
    for i in range(len(accept)):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0])
        if rank > 0:
            worst.append({
                "sample_id": accept[i]["sample_id"],
                "caption": accept[i]["caption"],
                "rank": rank,
                "top_distractor": accept[int(order[0])]["caption"],
                "subgroup": accept[i]["subgroup"],
            })
    worst.sort(key=lambda x: -x["rank"])
    failures = worst[:10]

    # ---------- 7. artifact registry + signed run record ----------
    artifacts_dir = runs_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_refs: Dict[str, Any] = {
        "manifest_id": manifest["manifest_id"],
        "split_id": splits["split_id"],
    }
    if adapter is not None:
        ckpt = artifacts_dir / f"{run_id}-lora.json"
        adapter.save(str(ckpt))
        artifact_refs["lora_checkpoint"] = {
            "path": str(ckpt),
            "sha256": _file_sha256(ckpt),
            "state_hash": adapter.state_hash(),
            "trainable_params": adapter.trainable_param_count(),
        }

    record: Dict[str, Any] = {
        "schema": "klcap-run-v1",
        "lab_version": __version__,
        "evaluator_version": EVALUATOR_VERSION,
        "run_id": run_id,
        "experiment_name": exp["name"],
        "role": exp.get("role", "candidate"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "config_hash": _sha256_obj(cfg),
        "seed": seed,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "encoder": {
            "text": getattr(txt_enc, "name", type(txt_enc).__name__),
            "image": getattr(img_enc, "name", type(img_enc).__name__),
            "dim": int(tr_txt.shape[1]),
        },
        "data": {
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest["records_sha256"],
            "split_id": splits["split_id"],
            "n_train": len(train),
            "n_validation": len(val),
            "n_acceptance": len(accept),
            "licences": manifest["licences"],
        },
        "metrics": metrics,
        "slices": slices,
        "fairness_verdict": fv,
        "telemetry": {**tel.to_dict(), "energy_per_query_joules": epq},
        "negative_log_summary": {
            "n_negatives_used": len(negative_logs),
            "schedule": schedule,
            "log": negative_logs[:500],
        },
        "loss_history_head": loss_history[:5],
        "loss_history_tail": loss_history[-5:] if loss_history else [],
        "failure_examples": failures,
        "artifacts": artifact_refs,
    }
    record["record_sha256"] = _sha256_obj({k: v for k, v in record.items()})

    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[pipeline] run saved -> {out}")
    print(
        f"[pipeline] recall@1={metrics['recall@1']:.3f} "
        f"probe={metrics['probe_accuracy']:.3f} "
        f"fairness_ok={fv['fairness_ok']} "
        f"energy={tel.energy_joules:.1f}J ({tel.energy_source})"
    )
    return record


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "no-git"
    except Exception:
        return "no-git"
