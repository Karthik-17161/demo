"""Command-line workflows for the Vision-Language Representation Lab.

Commands
--------
make-demo-data   Generate a licensed synthetic dataset + manifest (Input box).
freeze-splits    Freeze leakage-resistant train/validation/acceptance splits.
run              Execute one experiment from a YAML config (O2 or O3).
compare          Fair-comparison gate + O2/O3 delta report.
acceptance       Independent acceptance harness → ACCEPT/REVISE/HOLD report.
negative-tests   Run the six required failure demonstrations.
serve            Start the FastAPI operator service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .acceptance.harness import run_acceptance
from .acceptance.negative_tests import run_all_negative_tests
from .comparison import ComparisonBlocked, compare_runs, load_run
from .data.ingest import build_manifest, generate_synthetic_records
from .data.splits import freeze_splits
from .pipeline import load_config, run_experiment


def cmd_make_demo_data(args: argparse.Namespace) -> int:
    records = generate_synthetic_records(args.n_pairs, seed=args.seed)
    manifest = build_manifest(records, Path(args.out), name=f"demo{args.n_pairs}")
    print(f"[ingest] {manifest['n_records']} pairs -> {args.out}")
    print(f"[ingest] manifest_id={manifest['manifest_id']} licences={manifest['licences']}")
    return 0


def cmd_freeze_splits(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.manifest)
    manifest, records = load_manifest_records(manifest_dir)
    splits = freeze_splits(
        manifest, records, out_dir=Path(args.out), seed=args.seed,
        manifest_dir=manifest_dir,
    )
    print(f"[splits] frozen split_id={splits['split_id']}")
    print(f"[splits] counts={splits['counts']} scene_counts={splits['scene_counts']}")
    return 0


def load_manifest_records(manifest_dir: Path):
    from .data.ingest import load_manifest

    return load_manifest(manifest_dir)


def cmd_run(args: argparse.Namespace) -> int:
    record = run_experiment(Path(args.config), Path(args.runs_dir))
    return 0 if record else 1


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_run(Path(args.baseline))
    candidate = load_run(Path(args.candidate))
    try:
        report = compare_runs(baseline, candidate)
    except ComparisonBlocked as exc:
        print(f"[compare] BLOCKED: {exc}")
        return 2
    out = Path(args.out) if args.out else Path("runs") / f"comparison-{candidate['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[compare] gate=PASS saved -> {out}")
    for k, v in report["metrics_delta"].items():
        print(f"  {k}: baseline={v['baseline']} candidate={v['candidate']} delta={v['delta']}")
    res = report["resource"]
    if res["resource_improvement_fraction"] is not None:
        print(f"  resource improvement: {res['resource_improvement_fraction']:+.1%}")
    print(f"  critical regression check: {report['critical_regression_check']['verdict']}")
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    cfg_dir = Path(args.manifest_dir)
    splits_path = Path(args.splits)
    report = run_acceptance(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        manifest_dir=cfg_dir,
        splits_path=splits_path,
        out_dir=Path(args.out),
    )
    print(f"[acceptance] overall={report['overall']}")
    return 0


def cmd_negative_tests(args: argparse.Namespace) -> int:
    report = run_all_negative_tests(out_path=Path(args.out) if args.out else None)
    for name, res in report["tests"].items():
        mark = "PASS" if res["pass"] else "FAIL"
        print(f"[negative-tests] {mark}  {name}")
    print(f"[negative-tests] all_pass={report['all_pass']}")
    return 0 if report["all_pass"] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("uvicorn not installed; pip install -r requirements.lock", file=sys.stderr)
        return 1
    uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    p.add_argument("--version", action="version", version=f"klcap-lab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("make-demo-data", help="generate licensed demo dataset + manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--n-pairs", type=int, default=1500)
    s.add_argument("--seed", type=int, default=167)
    s.set_defaults(func=cmd_make_demo_data)

    s = sub.add_parser("freeze-splits", help="freeze leakage-resistant splits")
    s.add_argument("--manifest", required=True, help="path to manifest dir")
    s.add_argument("--out", required=True, help="output dir for splits.json")
    s.add_argument("--seed", type=int, default=167)
    s.set_defaults(func=cmd_freeze_splits)

    s = sub.add_parser("run", help="execute one experiment config")
    s.add_argument("--config", required=True)
    s.add_argument("--runs-dir", default="runs")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("compare", help="fair-comparison of two runs")
    s.add_argument("--baseline", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("acceptance", help="independent acceptance harness")
    s.add_argument("--baseline", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--manifest-dir", required=True)
    s.add_argument("--splits", required=True)
    s.add_argument("--out", default="evidence/acceptance")
    s.set_defaults(func=cmd_acceptance)

    s = sub.add_parser("negative-tests", help="six required failure demonstrations")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_negative_tests)

    s = sub.add_parser("serve", help="start the FastAPI operator service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
