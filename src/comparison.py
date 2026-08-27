"""O2 vs O3 comparison under the fair-comparison gate.

A comparison is *blocked* unless both runs used:
  • the identical acceptance manifest (same manifest_id + records hash);
  • the identical frozen split;
  • the identical evaluator version;
  • telemetry that is complete on both sides;
  • and the candidate stayed inside its approved compute envelope
    (trainable-parameter budget + FLOPs ceiling).

Negative test #1 (unequal data/compute) and #6 (missing telemetry) are
enforced here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation.fairness import critical_regression_check


class ComparisonBlocked(RuntimeError):
    """Raised when a baseline/candidate pair may not be compared."""


def load_run(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_runs(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    max_flops_ratio_ceiling: float = 1.10,
) -> Dict[str, Any]:
    """Fair-comparison gate + quality/fairness deltas. Raises if blocked."""
    blocks: List[str] = []

    b_data, c_data = baseline.get("data", {}), candidate.get("data", {})
    if b_data.get("manifest_id") != c_data.get("manifest_id"):
        blocks.append(
            f"unequal data: manifest {b_data.get('manifest_id')} != {c_data.get('manifest_id')}"
        )
    if b_data.get("manifest_sha256") != c_data.get("manifest_sha256"):
        blocks.append("unequal data: manifest content hash differs")
    if b_data.get("split_id") != c_data.get("split_id"):
        blocks.append(
            f"unequal splits: {b_data.get('split_id')} != {c_data.get('split_id')}"
        )

    if baseline.get("evaluator_version") != candidate.get("evaluator_version"):
        blocks.append("unequal evaluator versions")

    b_tel = baseline.get("telemetry", {})
    c_tel = candidate.get("telemetry", {})
    if not b_tel.get("complete") or not c_tel.get("complete"):
        blocks.append("telemetry unavailable/incomplete: resource claims disabled")

    # compute envelope: candidate FLOPs must stay within ceiling of baseline
    if b_tel.get("flops_train") and c_tel.get("flops_train"):
        ratio = c_tel["flops_train"] / max(b_tel["flops_train"], 1e-9)
        if ratio > max_flops_ratio_ceiling:
            blocks.append(
                f"compute envelope exceeded: candidate/baseline train-FLOPs ratio "
                f"{ratio:.2f} > {max_flops_ratio_ceiling:.2f}"
            )

    cand_cfg = candidate.get("config", {}).get("lora", {})
    if cand_cfg.get("enabled") and candidate.get("artifacts", {}).get("lora_checkpoint"):
        budget = int(cand_cfg.get("max_trainable_params", 100_000))
        used = int(candidate["artifacts"]["lora_checkpoint"].get("trainable_params", 0))
        if used > budget:
            blocks.append(f"parameter budget exceeded: {used} > {budget}")

    if blocks:
        raise ComparisonBlocked("; ".join(blocks))

    bm, cm = baseline["metrics"], candidate["metrics"]
    b_slices, c_slices = baseline["slices"], candidate["slices"]

    resource_improvement = None
    if b_tel.get("energy_joules") and c_tel.get("energy_joules"):
        resource_improvement = 1.0 - (
            c_tel["energy_joules"] / max(b_tel["energy_joules"], 1e-9)
        )

    regression = critical_regression_check(b_slices, c_slices)

    return {
        "schema": "klcap-comparison-v1",
        "baseline_run": baseline.get("run_id"),
        "candidate_run": candidate.get("run_id"),
        "gate": "PASS",
        "metrics_delta": {
            key: {
                "baseline": bm.get(key),
                "candidate": cm.get(key),
                "delta": (cm.get(key) - bm.get(key))
                if isinstance(bm.get(key), (int, float)) and isinstance(cm.get(key), (int, float))
                else None,
            }
            for key in ("recall@1", "recall@5", "recall@10", "mrr", "probe_accuracy")
        },
        "resource": {
            "baseline_energy_joules": b_tel.get("energy_joules"),
            "candidate_energy_joules": c_tel.get("energy_joules"),
            "baseline_flops_train": b_tel.get("flops_train"),
            "candidate_flops_train": c_tel.get("flops_train"),
            "resource_improvement_fraction": resource_improvement,
        },
        "critical_regression_check": regression,
    }
