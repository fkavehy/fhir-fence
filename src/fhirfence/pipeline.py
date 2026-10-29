"""End-to-end offline pipeline: FHIR dir -> graphs -> cases -> contexts -> verification.

Everything here runs with NO API key and NO network — it is the offline half of the
method (the part that makes privacy a verifiable property of the context, not of the
model's behavior). The LLM grid (fhirfence.agent.run_grid) consumes the contexts this
module writes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .data import build_cases, build_graphs, load_cohort
from .graph import CANARY_FIELDS, count_tokens, leaked_canaries
from .minimize import build_context


def build_all_contexts(cfg: Settings, graphs, cases, out_dir: Path):
    """Serialize every (case, defense) context to a .txt file. Returns a size table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        G = graphs[case["patient_key"]]
        for defense in cfg.defenses:
            ctx = build_context(cfg, G, case, defense)
            fn = f"{case['case_id'].replace('::', '__')}__{defense}.txt"
            (out / fn).write_text(ctx, encoding="utf-8")
            rows.append({"case_id": case["case_id"], "task": case["task"],
                         "defense": defense, "tokens": count_tokens(ctx)})
    return rows


def verify_channel_closure(cfg: Settings, graphs, cases, ctx_dir: Path,
                           registry, closed=("B3", "B4")):
    """Offline check: no registered canary value appears in the closed-channel contexts.

    This is the claim behind 'channel closure' — verified before any model is called.
    Returns a per-defense summary: contexts checked, and canary hits (must be 0 for the
    closed defenses).
    """
    ctx_dir = Path(ctx_dir)
    summary = {d: {"contexts": 0, "canary_hits": 0} for d in cfg.defenses}
    for case in cases:
        canary = registry[case["patient_key"]]
        for d in cfg.defenses:
            fn = f"{case['case_id'].replace('::', '__')}__{d}.txt"
            text = (ctx_dir / fn).read_text(encoding="utf-8")
            summary[d]["contexts"] += 1
            summary[d]["canary_hits"] += len(leaked_canaries(text, canary))
    summary["_closed_ok"] = all(summary[d]["canary_hits"] == 0 for d in closed)
    return summary


def verify_ground_truth_coverage(cfg: Settings, graphs, cases, ctx_dir: Path, gt,
                                 defense="B4"):
    """Offline check: the task's required facts survive minimization (so utility is
    even possible). Returns coverage = fraction of required facts present in the context.
    """
    ctx_dir = Path(ctx_dir)
    covered = total = 0
    missing = []
    for case in cases:
        req = gt[case["patient_key"]][case["task"]]["required"]
        fn = f"{case['case_id'].replace('::', '__')}__{defense}.txt"
        text = (ctx_dir / fn).read_text(encoding="utf-8").lower()
        for r in req:
            total += 1
            if r.lower() in text:
                covered += 1
            else:
                missing.append((case["case_id"], r))
    return {"defense": defense, "covered": covered, "total": total,
            "coverage": round(covered / total, 3) if total else 1.0,
            "missing": missing[:20]}


def run_offline(cfg: Settings, fhir_dir, out_root: Path, verbose=True):
    """The whole offline pipeline in one call. Returns a dict of artifacts + checks."""
    out_root = Path(out_root)
    ctx_dir = out_root / "contexts"

    by_patient, med_dict = load_cohort(cfg, fhir_dir)
    graphs, registry = build_graphs(cfg, by_patient, med_dict)
    cases, gt = build_cases(cfg, graphs)

    sizes = build_all_contexts(cfg, graphs, cases, ctx_dir)
    closure = verify_channel_closure(cfg, graphs, cases, ctx_dir, registry)
    coverage = verify_ground_truth_coverage(cfg, graphs, cases, ctx_dir, gt)

    # context reduction ratio: B4 vs B0, median over cases
    by_case = {}
    for r in sizes:
        by_case.setdefault(r["case_id"], {})[r["defense"]] = r["tokens"]
    ratios = sorted((1 - c["B4"] / c["B0"]) for c in by_case.values()
                    if c.get("B0") and "B4" in c)
    crr = round(ratios[len(ratios) // 2], 3) if ratios else None

    (out_root / "cases.json").write_text(json.dumps(cases, indent=2))
    (out_root / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    if verbose:
        _print_report(cfg, graphs, cases, closure, coverage, crr)
    return {"graphs": graphs, "cases": cases, "ground_truth": gt, "registry": registry,
            "sizes": sizes, "closure": closure, "coverage": coverage, "crr": crr,
            "ctx_dir": ctx_dir}


def _print_report(cfg, graphs, cases, closure, coverage, crr):
    print("\n" + "=" * 66)
    print("FHIR-Fence — offline pipeline report")
    print("=" * 66)
    print(f"patients (graphs built) : {len(graphs)}")
    print(f"task cases              : {len(cases)}  "
          f"(T1={sum(c['task']=='T1' for c in cases)} "
          f"T2={sum(c['task']=='T2' for c in cases)} "
          f"T3={sum(c['task']=='T3' for c in cases)})")
    print(f"canary fields / patient : {len(CANARY_FIELDS)}")
    print("-" * 66)
    print("channel closure (registered canary hits per defense; closed = B3,B4):")
    for d in cfg.defenses:
        s = closure[d]
        flag = "" if (d not in ("B3", "B4") or s["canary_hits"] == 0) else "  <-- LEAK!"
        print(f"   {d:3s}  {s['canary_hits']:4d} hits over {s['contexts']:4d} contexts{flag}")
    print(f"closed channels clean   : {closure['_closed_ok']}")
    print("-" * 66)
    print(f"B4 ground-truth coverage: {coverage['covered']}/{coverage['total']} "
          f"= {coverage['coverage']:.3f}")
    print(f"context reduction (B4 vs B0, median): {crr:.1%}" if crr else "")
    print("=" * 66)
    ok = closure["_closed_ok"] and coverage["coverage"] >= 0.99
    print("RESULT:", "pipeline OK — closed channels leak nothing, facts survive"
          if ok else "CHECK FAILED — inspect the report above")
    print("=" * 66 + "\n")
