"""Data-layer tests: manifest integrity, quality gates, leakage resistance."""

import copy
import json
from pathlib import Path

import pytest

from src.data.ingest import (
    build_manifest,
    generate_synthetic_records,
    load_manifest,
    validate_records,
)
from src.data.splits import check_leakage, freeze_splits, load_splits


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    records = generate_synthetic_records(300, seed=167)
    build_manifest(records, d, name="test")
    return d


def test_generated_records_pass_validation():
    records = generate_synthetic_records(200, seed=167)
    assert validate_records(records) == []


def test_manifest_roundtrip_verifies_hash(demo_dir):
    manifest, records = load_manifest(demo_dir)
    assert manifest["n_records"] == len(records)
    assert manifest["licences"] == ["CC-BY-4.0"]
    assert manifest["split_unit"] == "scene_id"


def test_tampered_records_refused(demo_dir):
    records_path = demo_dir / "records.jsonl"
    original = records_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    rec = json.loads(lines[0])
    rec["caption"] = "tampered caption text"
    lines[0] = json.dumps(rec, sort_keys=True)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        load_manifest(demo_dir)
    records_path.write_text(original, encoding="utf-8")


def test_unapproved_licence_flagged():
    records = generate_synthetic_records(20, seed=1)
    records[0]["licence"] = "Custom-EULA"
    problems = validate_records(records)
    assert any("unapproved licence" in p for p in problems)


def test_duplicate_sample_id_flagged():
    records = generate_synthetic_records(20, seed=1)
    records.append(copy.deepcopy(records[0]))
    problems = validate_records(records)
    assert any("duplicate sample_id" in p for p in problems)


def test_scene_split_no_leakage():
    records = generate_synthetic_records(300, seed=167)
    manifest = {"manifest_id": "m", "split_frozen": False}
    assigned = copy.deepcopy(records)   # freeze_splits stamps r["split"] on these
    splits = freeze_splits(manifest, assigned, out_dir=Path("evidence/_t"), seed=167)
    assert check_leakage(assigned, splits) == []
    counts = splits["counts"]
    assert counts["acceptance"] > 0 and counts["validation"] > 0
    # scene unit: every scene lives entirely inside one split
    scene_splits = {}
    for r in assigned:
        scene_splits.setdefault(r["scene_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in scene_splits.values())


def test_frozen_split_is_deterministic():
    records = generate_synthetic_records(120, seed=167)
    m1 = {"manifest_id": "m", "split_frozen": False}
    s1 = freeze_splits(m1, copy.deepcopy(records), out_dir=Path("evidence/_t1"), seed=167)
    m2 = {"manifest_id": "m", "split_frozen": False}
    s2 = freeze_splits(m2, copy.deepcopy(records), out_dir=Path("evidence/_t2"), seed=167)
    assert s1["assignments"] == s2["assignments"]
    assert s1["split_id"] == s2["split_id"]


def test_load_splits_roundtrip():
    records = generate_synthetic_records(60, seed=167)
    m = {"manifest_id": "m", "split_frozen": False}
    freeze_splits(m, copy.deepcopy(records), out_dir=Path("evidence/_t3"), seed=167)
    loaded = load_splits(Path("evidence/_t3/splits.json"))
    assert set(loaded["counts"]) == {"train", "validation", "acceptance"}


def test_split_assignment_tampering_is_refused(tmp_path):
    records = generate_synthetic_records(60, seed=167)
    manifest = {"manifest_id": "m", "split_frozen": False, "records_sha256": "records-hash"}
    out = tmp_path / "splits"
    freeze_splits(manifest, records, out_dir=out, seed=167)
    path = out / "splits.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_id = next(iter(payload["assignments"]))
    payload["assignments"][sample_id] = (
        "validation" if payload["assignments"][sample_id] != "validation" else "train"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split assignments hash"):
        load_splits(path)


def test_freeze_marks_the_actual_manifest_directory(tmp_path):
    records = generate_synthetic_records(60, seed=167)
    manifest_dir = tmp_path / "demo"
    manifest = build_manifest(records, manifest_dir, name="freeze")
    freeze_splits(manifest, records, tmp_path / "splits", manifest_dir=manifest_dir)
    saved = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["split_frozen"] is True
