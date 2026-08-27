"""Inspect the latest run record: metrics, fairness verdict, slices."""
import glob
import json
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/o3c-lora-hn-fair-*.json"
runs = sorted(glob.glob(pattern))
if not runs:
    print("no runs match", pattern)
    raise SystemExit(1)
rec = json.load(open(runs[-1], encoding="utf-8"))
print("run:", runs[-1])
print("recall@1:", rec["metrics"]["recall@1"])
print("fairness:", rec["fairness_verdict"])
print("-- by_subgroup --")
for k, v in rec["slices"]["by_subgroup"].items():
    print(f"  {k}: n={v['n']} r@1={v['recall@1']:.3f}")
print("-- by_domain --")
for k, v in rec["slices"]["by_domain"].items():
    print(f"  {k}: n={v['n']} r@1={v['recall@1']:.3f}")
