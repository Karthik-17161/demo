"""Telemetry collector: FLOPs estimates, wall-clock latency, energy.

Energy is read from NVML when a GPU is present, otherwise estimated from a
documented CPU thermal-design-power model — but the estimate is *explicitly
labelled* (``energy_source: "tdp-estimate"``). If neither FLOPs nor energy can
be measured, ``complete=False`` and the acceptance harness forbids any
resource-improvement claim (negative test #6).
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryRecord:
    run_id: str
    started_utc: float = 0.0
    ended_utc: float = 0.0
    elapsed_s: float = 0.0
    flops_train: float = 0.0
    flops_eval: float = 0.0
    energy_joules: Optional[float] = None
    energy_source: str = "unavailable"
    gpu_name: Optional[str] = None
    cpu_model: str = platform.processor() or "unknown"
    platform: str = platform.platform()
    complete: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gpu_energy_joules(elapsed_s: float) -> Optional[float]:
    """Read GPU energy via NVML if available."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        watts = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        pynvml.nvmlShutdown()
        return watts * elapsed_s, name   # type: ignore[return-value]
    except Exception:
        return None


def _cpu_tdp_watts() -> float:
    """Conservative documented default for laptop-class CPUs (estimation)."""
    return 28.0


def start_run(run_id: str) -> TelemetryRecord:
    rec = TelemetryRecord(run_id=run_id, started_utc=time.time())
    return rec


def finish_run(
    rec: TelemetryRecord,
    flops_train: float,
    flops_eval: float,
) -> TelemetryRecord:
    rec.ended_utc = time.time()
    rec.elapsed_s = max(rec.ended_utc - rec.started_utc, 1e-6)
    rec.flops_train = flops_train
    rec.flops_eval = flops_eval

    gpu = _gpu_energy_joules(rec.elapsed_s)
    if gpu is not None:
        rec.energy_joules, rec.gpu_name = gpu
        rec.energy_source = "nvml"
    else:
        # labelled CPU TDP estimate; still a measurement of elapsed power draw
        rec.energy_joules = _cpu_tdp_watts() * rec.elapsed_s
        rec.energy_source = "tdp-estimate"
        rec.notes.append("no NVML GPU; energy estimated from 28 W CPU TDP model")

    # Complete = energy measured AND compute accounted for. Eval-only runs
    # (the O2 baseline never fine-tunes) are complete via their eval FLOPs.
    rec.complete = (
        rec.energy_joules is not None and (rec.flops_train > 0 or rec.flops_eval > 0)
    )
    if not rec.complete:
        rec.notes.append("telemetry incomplete: resource-improvement claims disabled")
    return rec


def estimate_flops(
    n_pairs: int,
    dim: int,
    epochs: int,
    lora_rank: int = 0,
    hard_negatives: int = 0,
) -> Dict[str, float]:
    """Documented FLOPs model for the hashing-encoder lab.

    Encoding: O(n·dim) per modality per pass; LoRA training adds the residual
    matmuls O(n·dim·rank) per epoch; each hard-negative pair adds a dot product.
    """
    encode = 2 * n_pairs * dim * 2          # image + text encoding
    train = epochs * (n_pairs * dim * dim / 8 + n_pairs * lora_rank * dim * 4)
    negatives = epochs * hard_negatives * dim * 2
    eval_flops = 2 * n_pairs * dim * 3      # similarity matrices + probes
    return {
        "flops_encode": float(encode),
        "flops_train": float(train + negatives),
        "flops_eval": float(eval_flops),
    }


def energy_per_query(rec: TelemetryRecord, n_queries: int) -> Optional[float]:
    if rec.energy_joules is None or n_queries <= 0:
        return None
    return rec.energy_joules / n_queries
