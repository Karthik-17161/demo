"""Diagnose failing queries: show top-1 neighbour vs gold."""
from pathlib import Path

import numpy as np

from src.data.ingest import load_manifest
from src.data.splits import load_splits
from src.models.encoders import build_encoder

manifest, records = load_manifest(Path("evidence/demo"))
splits = load_splits(Path("evidence/splits/splits.json"))
for r in records:
    r["split"] = splits["assignments"][r["sample_id"]]

accept = [r for r in records if r["split"] == "acceptance"]
enc = build_encoder({"encoder": {"seed": 167}})
txt = enc["text"].encode_texts([r["caption"] for r in accept])
img = enc["image"].encode_images(
    [r["image_uri"] for r in accept], [r["scene_id"] for r in accept]
)
sims = txt @ img.T

fails = 0
for i, r in enumerate(accept):
    order = np.argsort(-sims[i])
    top1 = int(order[0])
    if top1 != i:
        fails += 1
        if fails <= 12:
            print(f"[{r['domain']}] Q : {r['caption']}")
            print(f"          G : {r['image_uri']}  sim={sims[i][i]:.3f}")
            print(f"          T1: {accept[top1]['image_uri']}  sim={sims[i][top1]:.3f}")
            print()
print("total fails:", fails, "/", len(accept))
