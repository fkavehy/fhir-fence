"""Attacked clinical agent: prompt assembly + checkpointed, resumable runner.

The model is reached only through the provider-agnostic `LLMClient` (fhirfence.llm):
this file contains no provider SDK import, so FHIR-Fence runs against Gemini, OpenAI,
Anthropic, an OpenAI-compatible endpoint, or the offline echo client, chosen entirely
by configs/config.yaml -> model.provider.

Prompts and attacks come from configs/prompts.yaml; model settings from config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .llm import LLMClient, make_llm_client


def build_prompt(cfg: Settings, case, defense, attack, ctx_root: Path):
    """Assemble the exact model-visible (system, user) prompt."""
    pr = cfg.prompts
    ctx = (Path(ctx_root) / f"{case['case_id'].replace('::', '__')}__{defense}.txt").read_text()
    a = pr["attacks"][attack]
    system = pr["system_base"]
    if defense == "B1":
        system = system + " " + pr["system_privacy_suffix"]

    payload = a.get("payload") or ""
    where = a.get("where")
    retrieved = payload if where == "retrieved" else ""
    question = case["question"] + (payload if where == "user" else "")
    user = f"PATIENT RECORD:\n{ctx}\n{retrieved}\n\nQUESTION:\n{question}"
    return system, user


def call_model(cfg: Settings, client: LLMClient, system, user):
    """One API call with retry/backoff. Returns (text, meta). Kept for backward
    compatibility; delegates to the LLMClient."""
    return client.generate(system, user)


def load_done(path: Path):
    done = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["run_id"]] = r
    return done


def run_grid(cfg: Settings, cases, ctx_root: Path, run_log: Path,
             defenses=None, attacks=None, label="", client: LLMClient | None = None):
    """Checkpointed, resumable, append-only. Re-run after any interruption; completed
    run_ids are skipped. Each run writes to its own log file (pass a distinct run_log).

    The model backend is whatever configs/config.yaml -> model.provider selects; pass an
    explicit `client` to override (e.g. an EchoClient in tests).
    """
    m = cfg.model
    defenses = defenses or cfg.defenses
    attacks = attacks or list(cfg.prompts["attacks"].keys())
    sleep = 60.0 / float(m.get("rpm_limit", 60)) + 0.5
    run_log = Path(run_log)
    run_log.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(run_log)
    todo = [(c, d, a) for c in cases for a in attacks for d in defenses
            if f"{c['case_id']}::{a}::{d}" not in done]
    print(f"[{label}] {len(todo)} calls to make ({len(done)} already logged) "
          f"~{len(todo) * sleep / 60:.0f} min")
    if not todo:
        return

    client = client or make_llm_client(m)
    with open(run_log, "a") as fh:
        for i, (c, d, a) in enumerate(todo, 1):
            run_id = f"{c['case_id']}::{a}::{d}"
            system, user = build_prompt(cfg, c, d, a, ctx_root)
            text, meta = client.generate(system, user)
            rec = {
                "run_id": run_id, "case_id": c["case_id"],
                "patient_key": c["patient_key"], "task": c["task"],
                "attack": a, "defense": d, "is_dev": c["is_dev"],
                "provider": m.get("provider"), "model": m["name"],
                "temperature": float(m.get("temperature", 0.0)),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system_prompt": system, "user_prompt": user,
                "context_sha256": hashlib.sha256(user.encode()).hexdigest()[:16],
                "response": text, **meta,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  [{label}] {i}/{len(todo)} last={run_id[:38]} "
                      f"err={meta['error'] is not None}")
            time.sleep(sleep)
