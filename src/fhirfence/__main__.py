"""Command-line interface.

    python -m fhirfence demo                         # offline, no key, synthetic data
    python -m fhirfence build   --fhir DIR --out OUT # graphs + contexts + verify (offline)
    python -m fhirfence run     --out OUT            # LLM grid (needs provider + key)
    python -m fhirfence evaluate --out OUT           # score ASR / utility / CRR + CIs

Every stage reads its parameters from configs/. Use --config to point elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings


def _cmd_demo(args):
    from .pipeline import run_offline
    here = Path(__file__).resolve().parents[2]
    fhir = Path(args.fhir) if args.fhir else here / "examples" / "synthetic_fhir"
    if not fhir.exists() or not any(fhir.glob("*.ndjson*")):
        print(f"No synthetic data at {fhir} — generating it...")
        sys.path.insert(0, str(here / "examples"))
        from make_synthetic_fhir import generate
        generate(args.n, fhir)
    cfg = load_settings(args.config)
    run_offline(cfg, fhir, Path(args.out))


def _cmd_build(args):
    from .pipeline import run_offline
    cfg = load_settings(args.config)
    run_offline(cfg, Path(args.fhir), Path(args.out))


def _cmd_run(args):
    from .agent import run_grid
    cfg = load_settings(args.config)
    out = Path(args.out)
    cases = json.loads((out / "cases.json").read_text())
    run_grid(cfg, cases, ctx_root=out / "contexts",
             run_log=out / "agent_runs.jsonl",
             defenses=args.defenses, label="grid")


def _cmd_evaluate(args):
    from .evaluate import load_runs, score, write_tables
    from .graph import build_canary_registry
    cfg = load_settings(args.config)
    out = Path(args.out)
    runs = load_runs(out / "agent_runs.jsonl")
    if runs.empty:
        print("No runs found. Run `python -m fhirfence run --out ...` first.")
        return
    gt = json.loads((out / "ground_truth.json").read_text())
    canaries = build_canary_registry(sorted(gt), cfg.canary_seed)
    results = score(cfg, runs, canaries, gt)
    write_tables(results, out / "tables")
    print("\nASR (attack success / leakage):\n", results["asr"])
    print("\nUtility:\n", results["utility"])
    print("\nBootstrap CIs:\n", results["bootstrap"].to_string(index=False))
    print(f"\nTables written to {out / 'tables'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fhirfence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to configs/ (default: auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="offline demo on synthetic data (no key)")
    p.add_argument("--out", default="demo_out")
    p.add_argument("--fhir", default=None)
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("build", help="build graphs + contexts + verify (offline)")
    p.add_argument("--fhir", required=True)
    p.add_argument("--out", default="out")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("run", help="run the LLM grid (needs provider + key)")
    p.add_argument("--out", default="out")
    p.add_argument("--defenses", nargs="*", default=None)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("evaluate", help="score ASR / utility / CRR + bootstrap CIs")
    p.add_argument("--out", default="out")
    p.set_defaults(func=_cmd_evaluate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
