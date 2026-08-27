"""Dataset ingestion: build a licence/provenance manifest from raw records.

A *record* is one licensed image-text pair:

    {
      "sample_id":  "demo-000001",
      "image_uri":  "images/outdoor/scene-0003/000007.jpg",
      "caption":    "a red bicycle leaning on a wooden fence, viewed from angle 2",
      "source":     "demo-synthetic-v1",
      "licence":    "CC-BY-4.0",
      "access_date":"2026-08-20",
      "scene_id":   "outdoor-scene-0003",   # independent unit for splitting
      "domain":     "outdoor",
      "subgroup":   "region-eu"             # approved audit slice
    }

The manifest records every field required by the architecture: source, licence,
access date, per-sample hash, domain, subgroup and (later) the frozen split.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REQUIRED_FIELDS = (
    "sample_id", "image_uri", "caption", "source",
    "licence", "access_date", "scene_id", "domain", "subgroup",
)

APPROVED_LICENCES = {"CC-BY-4.0", "CC0-1.0", "CC-BY-SA-4.0", "ODbL-1.0"}

# Approved audit axes — every metric is reported per domain and per subgroup.
APPROVED_DOMAINS = ["indoor", "outdoor"]
APPROVED_SUBGROUPS = [
    "region-eu", "region-na", "region-asia",
    "lighting-low", "lighting-normal",
    "rare-object", "common-object",
]

# (object_class, [(caption, uri_slug), ...]) — each variant becomes its own
# scene instance. Variants of one class share the key noun but differ in
# attribute words (colour / place), so same-class/different-scene pairs are
# the classic confusable-but-distinguishable hard negatives. Slugs mirror
# real-world datasets whose filenames encode image content, giving the
# hashing encoders genuine cross-modal vocabulary overlap.
_SCENES = {
    "indoor": [
        ("mug", [
            ("a red mug on a kitchen counter", "red-mug-kitchen-counter"),
            ("a blue mug on an office desk", "blue-mug-office-desk"),
        ]),
        ("cat", [
            ("a sleeping cat on a sofa cushion", "sleeping-cat-sofa-cushion"),
            ("a playful cat on a windowsill", "playful-cat-windowsill"),
        ]),
        ("clock", [
            ("an antique clock above a stone fireplace", "antique-clock-stone-fireplace"),
            ("a wooden clock in a hallway", "wooden-clock-hallway"),
        ]),
        ("books", [
            ("a stack of paperbacks beside a reading lamp", "paperbacks-reading-lamp"),
            ("a row of novels on a study shelf", "novels-study-shelf"),
        ]),
        ("violin", [
            ("a violin resting inside an open case", "violin-open-case"),
            ("a violin hanging on a studio wall", "violin-studio-wall"),
        ]),
    ],
    "outdoor": [
        ("bicycle", [
            ("a red bicycle leaning on a wooden fence", "red-bicycle-wooden-fence"),
            ("a blue bicycle parked at a market square", "blue-bicycle-market-square"),
        ]),
        ("bridge", [
            ("a narrow canal crossed by a stone bridge", "canal-stone-bridge"),
            ("a wide river crossed by an iron bridge", "river-iron-bridge"),
        ]),
        ("balloon", [
            ("a hot air balloon rising over farmland", "hot-air-balloon-farmland"),
            ("a hot air balloon drifting above a valley", "hot-air-balloon-valley"),
        ]),
        ("truck", [
            ("a food truck parked at a beach boardwalk", "food-truck-beach-boardwalk"),
            ("a food truck stationed at a city park", "food-truck-city-park"),
        ]),
        ("lighthouse", [
            ("a lighthouse on a rocky cliff at dusk", "lighthouse-rocky-cliff-dusk"),
            ("a lighthouse above a sandy cove at dawn", "lighthouse-sandy-cove-dawn"),
        ]),
    ],
}

# rare-object classes feed the approved subgroup axis
_RARE_CLASSES = {"clock", "violin", "balloon", "lighthouse"}

_REGIONS = ["region-eu", "region-na", "region-asia"]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_synthetic_records(n_pairs: int, seed: int = 167) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic image-text pairs with scene structure.

    Scenes group several near-duplicate captions so that split-by-scene is
    meaningful and random row splits would leak.
    """
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    sid = 0
    scene_no = 0
    while len(records) < n_pairs:
        domain = "indoor" if scene_no % 2 == 0 else "outdoor"
        classes = _SCENES[domain]
        obj_class = classes[(scene_no // 2) % len(classes)][0]
        variants = next(v for c, v in classes if c == obj_class)
        caption, slug = variants[(scene_no // (2 * len(classes))) % len(variants)]
        # viewpoint tags mirror real capture metadata stored in filenames;
        # angles are UNIQUE within a scene so same-scene near-duplicates stay
        # cross-modally distinguishable (no identical-caption coin flips)
        angles = rng.sample(range(1, 10), 4)
        n_rows = rng.randint(2, 4)   # 2–4 rows share one scene
        for angle in angles[:n_rows]:
            if len(records) >= n_pairs:
                break
            sid += 1
            variant = f"{caption}, viewed from angle {angle}"
            lighting = "lighting-low" if rng.random() < 0.18 else "lighting-normal"
            records.append({
                "sample_id": f"demo-{sid:06d}",
                "image_uri": f"images/{domain}/{slug}/angle-{angle}/scene-{scene_no:04d}/{sid:06d}.jpg",
                "caption": variant,
                "source": "demo-synthetic-v1",
                "licence": "CC-BY-4.0",
                "access_date": "2026-08-20",
                "scene_id": f"{domain}-scene-{scene_no:04d}",
                "domain": domain,
                "object_class": obj_class,
                "subgroup": (
                    "rare-object" if obj_class in _RARE_CLASSES else "common-object"
                ) if rng.random() < 0.34 else rng.choice([rng.choice(_REGIONS), lighting]),
            })
        scene_no += 1
    return records


def build_manifest(records: Iterable[Dict[str, Any]], out_dir: Path, name: str = "demo") -> Dict[str, Any]:
    """Validate records and write ``manifest.json`` + ``records.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = [dict(r) for r in records]
    problems = validate_records(recs)
    if problems:
        raise ValueError(f"record validation failed: {problems[:5]}")

    for r in recs:
        r["content_hash"] = _hash_text(r["caption"] + "|" + r["image_uri"])

    manifest: Dict[str, Any] = {
        "manifest_id": f"manifest-{name}-{_hash_text(json.dumps([r['sample_id'] for r in recs]))[:12]}",
        "created_utc": _utc_now(),
        "n_records": len(recs),
        "licences": sorted({r["licence"] for r in recs}),
        "sources": sorted({r["source"] for r in recs}),
        "domains": sorted({r["domain"] for r in recs}),
        "subgroups": sorted({r["subgroup"] for r in recs}),
        "approved_domains": sorted(APPROVED_DOMAINS),
        "approved_subgroups": sorted(APPROVED_SUBGROUPS),
        "split_unit": "scene_id",
        "split_frozen": False,
        "records_path": "records.jsonl",
        "records_sha256": None,
    }
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    manifest["records_sha256"] = _file_sha256(records_path)

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def load_manifest(manifest_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load manifest + records, verifying the records hash (artifact integrity)."""
    manifest = json.loads((Path(manifest_dir) / "manifest.json").read_text(encoding="utf-8"))
    records_path = Path(manifest_dir) / manifest.get("records_path", "records.jsonl")
    actual = _file_sha256(records_path)
    expected = manifest.get("records_sha256")
    if expected and actual != expected:
        raise ValueError(
            f"artifact mismatch: records hash {actual} != manifest hash {expected}"
        )
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, records


def validate_records(records: List[Dict[str, Any]]) -> List[str]:
    """Quality checks: required fields, licences, domains, subgroups, duplicates."""
    problems: List[str] = []
    seen_ids: set = set()
    seen_caption_image: set = set()
    for i, r in enumerate(records):
        missing = [f for f in REQUIRED_FIELDS if not r.get(f)]
        if missing:
            problems.append(f"row {i}: missing fields {missing}")
            continue
        if r["licence"] not in APPROVED_LICENCES:
            problems.append(f"row {i}: unapproved licence {r['licence']}")
        if r["domain"] not in APPROVED_DOMAINS:
            problems.append(f"row {i}: unapproved domain {r['domain']}")
        if r["subgroup"] not in APPROVED_SUBGROUPS:
            problems.append(f"row {i}: unapproved subgroup {r['subgroup']}")
        if r["sample_id"] in seen_ids:
            problems.append(f"row {i}: duplicate sample_id {r['sample_id']}")
        seen_ids.add(r["sample_id"])
        key = (r["caption"], r["image_uri"])
        if key in seen_caption_image:
            problems.append(f"row {i}: exact duplicate caption+image pair")
        seen_caption_image.add(key)
        if len(r["caption"].split()) < 3:
            problems.append(f"row {i}: caption too short")
    return problems


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
