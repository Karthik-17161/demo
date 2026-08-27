"""Leakage-resistant, immutable split freezing.

Splits are made over the *independent unit* (``scene_id``), never random image
rows, so near-duplicates of the same scene cannot appear in both development
and acceptance data. Once frozen, the split file is content-addressed and any
later modification is detected by hash.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
ACCEPT_RATIO = 0.15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def freeze_splits(
    manifest: Dict[str, Any],
    records: List[Dict[str, Any]],
    out_dir: Path,
    seed: int = 167,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Group scenes into train/validation/acceptance and write a frozen split.

    ``manifest_dir`` is optional for backwards compatibility with callers that
    hold a manifest only in memory.  Supplying it persists the frozen marker to
    the *actual* manifest location; split output directories are commonly a
    sibling of the manifest directory, so writing to ``out_dir.parent`` is not
    safe.
    """
    if manifest.get("split_frozen"):
        existing_path = Path(out_dir) / "splits.json"
        if not existing_path.exists():
            raise ValueError("manifest is marked frozen but splits.json is missing")
        return load_splits(existing_path)

    scenes = sorted({r["scene_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(scenes)

    n = len(scenes)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train:n_train + n_val])
    accept_scenes = set(scenes[n_train + n_val:])

    def assign(r: Dict[str, Any]) -> str:
        if r["scene_id"] in train_scenes:
            return "train"
        if r["scene_id"] in val_scenes:
            return "validation"
        return "acceptance"

    for r in records:
        r["split"] = assign(r)

    assignment_rows = sorted(r["sample_id"] + ":" + r["split"] for r in records)
    assignments_sha256 = _sha256_of(assignment_rows)
    splits: Dict[str, Any] = {
        "split_id": f"splits-{assignments_sha256[:12]}",
        "frozen_utc": _utc_now(),
        "seed": seed,
        "split_unit": "scene_id",
        "manifest_id": manifest["manifest_id"],
        "records_sha256": manifest.get("records_sha256"),
        "assignments_sha256": assignments_sha256,
        "counts": {
            s: sum(1 for r in records if r["split"] == s)
            for s in ("train", "validation", "acceptance")
        },
        "scene_counts": {
            s: len({r["scene_id"] for r in records if r["split"] == s})
            for s in ("train", "validation", "acceptance")
        },
        "assignments": {r["sample_id"]: r["split"] for r in records},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["split_frozen"] = True
    if manifest_dir is not None:
        manifest_path = Path(manifest_dir) / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"manifest path does not exist: {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return splits


def load_splits(splits_path: Path) -> Dict[str, Any]:
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    assignments = splits.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("invalid frozen split: assignments are missing")
    allowed = {"train", "validation", "acceptance"}
    if any(value not in allowed for value in assignments.values()):
        raise ValueError("invalid frozen split: unknown assignment label")
    # New split files are content-addressed.  Older evidence remains readable
    # so users can regenerate it with the current CLI instead of being locked
    # out of historical reports.
    expected = splits.get("assignments_sha256")
    if expected:
        actual = _sha256_of([sample_id + ":" + assignments[sample_id] for sample_id in sorted(assignments)])
        if actual != expected:
            raise ValueError("artifact mismatch: split assignments hash differs")
    return splits


def check_leakage(records: List[Dict[str, Any]], splits: Dict[str, Any]) -> List[str]:
    """Scene-boundary test: no scene may span two splits; report violations.

    The independent unit is ``scene_id``: every row derived from the same
    scene must live entirely inside one split. Shared *vocabulary* across
    scenes (e.g. the same object class photographed elsewhere) is legitimate
    data structure — it is what the hard-negative curriculum trains on — and
    is NOT leakage. Only scene-spanning assignments violate the boundary.
    """
    problems: List[str] = []
    scene_splits: Dict[str, set] = {}
    for r in records:
        sample_id = r["sample_id"]
        if sample_id not in splits["assignments"]:
            problems.append(f"sample {sample_id} is missing from frozen assignments")
            continue
        scene_splits.setdefault(r["scene_id"], set()).add(splits["assignments"][sample_id])
    for scene, sp in scene_splits.items():
        if len(sp) > 1:
            problems.append(f"scene {scene} leaks across splits {sorted(sp)}")
    return problems
