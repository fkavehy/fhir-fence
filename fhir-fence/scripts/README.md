# Scripts

The pipeline is driven entirely by the CLI — no separate scripts are needed:

```bash
python -m fhirfence demo                        # offline demo on synthetic data (no key)
python -m fhirfence build   --fhir DIR --out OUT # stages 1–3 + verification (offline)
python -m fhirfence run     --out OUT            # stage 4: LLM grid (needs provider + key)
python -m fhirfence evaluate --out OUT           # stage 5: ASR / utility / CRR + CIs
```

| Stage | What | Uses | API key |
|------:|------|------|:---:|
| 1 | load FHIR + cohort        | `data.load_cohort`          | no  |
| 2 | build graphs + canaries   | `data.build_graphs`         | no  |
| 3 | build contexts (B0–B4)    | `minimize.build_context`    | no  |
| – | **verify** closure + coverage | `pipeline.run_offline`  | no  |
| 4 | run agent grid            | `agent.run_grid`            | **yes** |
| 5 | evaluate                  | `evaluate.score`            | no  |
| 6 | PHI audit (optional)      | Presidio scan of contexts   | no  |

Stage 4 needs whatever provider + key you set in `configs/config.yaml → model`
(`gemini`, `openai`, `anthropic`, or `echo` for a no-key offline dry run). See the
User manual in the top-level `README.md`.

The original Colab notebooks (00–06) that produced the paper are kept under
`notebooks/` for reference; the CLI above is the clean, config-driven, provider-agnostic
equivalent.
