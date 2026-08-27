"""Hard-negative curriculum for contrastive vision-language training.

Curriculum stages (logged exactly, per run, which negatives were used):

* stage ``easy``   — random in-batch negatives (default InfoNCE behaviour);
* stage ``medium`` — negatives sharing the domain (indoor vs outdoor);
* stage ``hard``   — semantically-close negatives: same object class but a
  different scene (e.g. "red bicycle at fence" vs "bicycle at canal"), the
  classic confusable pairs.

The sampler is deterministic given ``(seed, stage)`` and emits an audit trail
``negative_log`` listing every negative pair index used, satisfying the
architecture requirement to "log exactly which negatives were used".
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

STAGES = ("easy", "medium", "hard")


@dataclass
class CurriculumResult:
    stage: str
    pair_indices: List[Tuple[int, int]] = field(default_factory=list)
    negative_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "n_pairs": len(self.pair_indices),
            "pair_indices": [list(p) for p in self.pair_indices],
            "negative_log": self.negative_log,
        }


def _hash_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]


def sample_curriculum(
    records: Sequence[Dict[str, Any]],
    stage: str,
    seed: int = 167,
    max_pairs: int = 256,
) -> CurriculumResult:
    """Return hard-negative query/negative indices for one training round.

    Indices refer to positions within the *training split* record list.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage '{stage}'; expected one of {STAGES}")

    rng = random.Random(f"{seed}:{stage}:{_hash_ids([r['sample_id'] for r in records])}")
    n = len(records)
    result = CurriculumResult(stage=stage)

    if stage == "easy":
        # random negatives: any different row
        for _ in range(min(max_pairs, n)):
            qi = rng.randrange(n)
            ni = rng.randrange(n)
            if ni == qi:
                continue
            result.pair_indices.append((qi, ni))
            result.negative_log.append({
                "query": records[qi]["sample_id"],
                "negative": records[ni]["sample_id"],
                "reason": "random",
            })
        return result

    if stage == "medium":
        # same-domain negatives: same indoor/outdoor world, different scene
        by_domain: Dict[str, List[int]] = {}
        for i, r in enumerate(records):
            by_domain.setdefault(r["domain"], []).append(i)
        domains = list(by_domain)
        for _ in range(min(max_pairs, n)):
            d = rng.choice(domains)
            pool = by_domain[d]
            if len(pool) < 2:
                continue
            qi, ni = rng.sample(pool, 2)
            if records[qi]["scene_id"] == records[ni]["scene_id"]:
                continue
            result.pair_indices.append((qi, ni))
            result.negative_log.append({
                "query": records[qi]["sample_id"],
                "negative": records[ni]["sample_id"],
                "reason": f"same-domain:{d}",
            })
        return result

    # hard: same object class, different scene (confusable but separable)
    def obj_class(r: Dict[str, Any]) -> str:
        return r.get("object_class") or r["caption"].split(", viewed from")[0]

    by_class: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_class.setdefault(obj_class(r), []).append(i)

    for base, pool in sorted(by_class.items()):
        if len(pool) < 2:
            continue
        scenes: Dict[str, int] = {}
        for i in pool:
            scenes.setdefault(records[i]["scene_id"], i)
        scene_ids = sorted(scenes)
        for a, b in zip(scene_ids, scene_ids[1:]):
            if len(result.pair_indices) >= max_pairs:
                break
            qi, ni = scenes[a], scenes[b]
            result.pair_indices.append((qi, ni))
            result.negative_log.append({
                "query": records[qi]["sample_id"],
                "negative": records[ni]["sample_id"],
                "reason": f"same-class-different-scene:{base[:40]}",
            })
    return result


def curriculum_schedule(total_rounds: int) -> List[str]:
    """Easy → medium → hard schedule across training rounds."""
    third = max(1, total_rounds // 3)
    schedule: List[str] = []
    for r in range(total_rounds):
        if r < third:
            schedule.append("easy")
        elif r < 2 * third:
            schedule.append("medium")
        else:
            schedule.append("hard")
    return schedule


def serialize_schedule(schedule: List[str]) -> str:
    return json.dumps(schedule)
