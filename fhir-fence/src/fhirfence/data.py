"""Load FHIR into patient graphs and build per-task cases.

This is the glue between a FHIR export on disk and the graph/minimizer. It replaces the
cohort/graph/policy notebook stages with a small, documented API so another team can run
FHIR-Fence on *their* FHIR data without touching the method code.

Expected input (`fhir_dir`): one directory of newline-delimited JSON (NDJSON), optionally
gzip-compressed (`.ndjson` or `.ndjson.gz`). Each line is one FHIR resource. A resource's
profile is read from `meta.profile[0]` (MIMIC-IV FHIR convention); if absent, the bare
`resourceType` is used. Resources link to their patient via `subject`/`patient` and to
their encounter via `encounter`/`context`/`partOf` — standard FHIR references.

The bundled synthetic sample under `examples/synthetic_fhir/` is in exactly this format,
so the same loader serves the demo and real data.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from .config import Settings
from .graph import (
    build_canary_registry,
    build_patient_graph,
    encounter_members,
    load_medication_dictionary,
)


def _profile_of(res: dict) -> str:
    prof = (res.get("meta", {}) or {}).get("profile") or []
    if prof:
        return prof[0].rsplit("/", 1)[-1]
    return res.get("resourceType", "Unknown")


def _iter_resources(fhir_dir: Path):
    for p in sorted(Path(fhir_dir).glob("*.ndjson*")):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _patient_key_of(res: dict) -> str | None:
    if res.get("resourceType") == "Patient":
        return res.get("id")
    for field in ("subject", "patient"):
        ref = (res.get(field) or {}).get("reference", "")
        if ref.startswith("Patient/"):
            return ref.split("/", 1)[1]
    return None


def load_cohort(cfg: Settings, fhir_dir, patient_keys=None):
    """Read a FHIR directory into per-patient, per-profile resource buckets.

    Returns (by_patient, med_dict) where
        by_patient[pk][profile] = [resource, ...]   (only in-scope profiles)
        med_dict[medication_id]  = Medication resource
    """
    in_scope = set(cfg.in_scope_profiles)
    want = set(patient_keys) if patient_keys else None
    by_patient: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for res in _iter_resources(fhir_dir):
        profile = _profile_of(res)
        if profile not in in_scope:
            continue
        pk = _patient_key_of(res)
        if pk is None or (want and pk not in want):
            continue
        by_patient[pk][profile].append(res)

    med_dict = load_medication_dictionary(fhir_dir)
    return {pk: dict(d) for pk, d in by_patient.items()}, med_dict


def build_graphs(cfg: Settings, by_patient, med_dict, seed=None):
    """Build one injected patient graph per patient. Returns {pk: networkx.DiGraph}."""
    seed = cfg.canary_seed if seed is None else seed
    registry = build_canary_registry(sorted(by_patient), seed)
    graphs = {}
    for pk, profiles in by_patient.items():
        if "MimicPatient" not in profiles:
            continue
        graphs[pk] = build_patient_graph(cfg, pk, profiles, registry[pk], med_dict)
    return graphs, registry


def _focal_encounter(G):
    """Pick the encounter with the most attached members (the richest encounter)."""
    best, best_n = None, -1
    for n, d in G.nodes(data=True):
        if d.get("resource_type") == "Encounter":
            n_members = len(encounter_members(G, n))
            if n_members > best_n:
                best, best_n = n, n_members
    return best


def _members_by_type(G, enc, rtype):
    return [n for n in encounter_members(G, enc)
            if G.nodes[n]["resource_type"] == rtype]


def build_cases(cfg: Settings, graphs, dev_keys=None):
    """Build the per-(patient, task) case list the agent/evaluator consume.

    A patient is task-eligible for a task if its focal encounter has at least one
    member of each required type. Returns (cases, ground_truth).

        case = {case_id, patient_key, task, question, encounter_node,
                authorized_types, patient_fields, is_dev}
        ground_truth[pk][task] = {"required": [exact strings the answer must contain]}
    """
    dev_keys = set(dev_keys or [])
    pol = cfg.policy["tasks"]
    cases, gt = [], defaultdict(dict)

    for pk, G in graphs.items():
        enc = _focal_encounter(G)
        if enc is None:
            continue
        meds = _members_by_type(G, enc, "Medication") + \
            [n for n in encounter_members(G, enc)
             if G.nodes[n]["resource_type"] in ("MedicationRequest", "MedicationAdministration")]
        conds = _members_by_type(G, enc, "Condition")
        obs = _members_by_type(G, enc, "Observation")

        task_members = {"T1": meds, "T2": conds, "T3": obs}
        for task, members in task_members.items():
            if not members:
                continue
            question = pol[task]["question"]
            target = None
            if task == "T3":
                target = _pick_t3_target(cfg, G, obs)
                if target is None:
                    continue
                question = question.replace("{target}", G.nodes[target]["label"])
                required = [_fmt_obs(G, target)]
            elif task == "T1":
                required = [G.nodes[m]["label"].split(" — ")[0] for m in members]
            else:  # T2
                required = [G.nodes[m]["label"] for m in members]

            gt[pk][task] = {"required": [r for r in required if r]}
            cases.append({
                "case_id": f"{pk}::{task}",
                "patient_key": pk, "task": task,
                "question": question,
                "encounter_node": enc,
                "authorized_types": pol[task]["authorized_types"],
                "patient_fields": pol[task]["patient_fields"],
                "is_dev": pk in dev_keys,
            })
    return cases, {pk: dict(d) for pk, d in gt.items()}


def _pick_t3_target(cfg, G, obs_nodes):
    whitelist = cfg.policy.get("t3_target_whitelist", [])
    exclude = [s.lower() for s in cfg.policy.get("t3_exclude_substrings", [])]
    for n in sorted(obs_nodes, key=lambda x: G.nodes[x].get("time") or "", reverse=True):
        lbl = str(G.nodes[n].get("label") or "")
        val = G.nodes[n].get("value")
        if val is None:
            continue
        if any(e in lbl.lower() for e in exclude):
            continue
        if any(w.lower() in lbl.lower() for w in whitelist):
            return n
    return None


def _fmt_obs(G, node):
    return str(G.nodes[node].get("value") or "")
