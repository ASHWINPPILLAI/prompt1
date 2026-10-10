# docs/architecture.md — System architecture reference.
# Paste into P0 prompt when starting each coding agent session.

# PolicyPilot: System Architecture

## Overview

One pipeline serves all prompt versions (V0–V6 and 'routed'). All differences
in evaluation results come from the prompt YAML, retrieval flag, ReAct flag,
and verifier flag — never from the underlying plumbing.

```
User question
    │
    ▼
[Question Router]  ─── OUT_OF_SCOPE / ADVERSARIAL ───► [Refusal Template]
    │
    ├── LOOKUP    ──► [V4 RAG + CoT pipeline]
    ├── CALC      ──► [V5 ReAct + Calculator]
    ├── FALLBACK  ──► [V6 ReAct + Verifier]
    └── (offline or direct) ──► [Prompt Version V0–V6]
                                      │
                                      ▼
                              [Retriever: FAISS + MiniLM]
                                      │
                              [Doc2Dial KB: SSA, VA, DMV]
                                      │
                              [LLM Client: OpenAI-compatible]
                                      │
                         ┌────────────┴────────────┐
                     [Generator]              [Judge / Verifier]
                         │
                         ▼
                    [RunRecord]
                         │
                    [Eval Harness]
                         │
                    [results/]
```

## Offline phase (once)

1. `make data` — downloads all 6 public datasets
2. `make ingest` — parses Doc2Dial docs → clauses.jsonl + chunks.jsonl
3. `make index` — embeds chunks with MiniLM → FAISS index

## Online phase (per query)

1. Retriever: embed query → FAISS top-k → list of {clause_id, text, score}
2. Prompt builder: load prompt YAML → render messages (system, few-shot, user)
3. LLM client: call with retry + cache → raw response
4. JSON parser: parse → repair once → RunRecord.parsed_output
5. [V5/V6] ReAct loop: Thought → Action → tool call → Observation → repeat
6. [V6] Verifier: claim-level faithfulness check → strip unsupported claims
7. RunRecord: save to results/runs/, log to usage_log.jsonl

## Repository layout

```
policypilot/
  AGENTS.md              Agent rules
  README.md              Setup and reproduction guide
  Makefile               All make targets
  requirements.txt       Dependencies
  .env.example           Key placeholders
  configs/
    default.yaml         Runtime settings (all annotated)
    sweeps.yaml          Experiment grids (E4, E5, E6)
  data/
    public/              Downloaded datasets (read-only after D1)
    processed/           clauses.jsonl, chunks.jsonl
    index/               FAISS files (git-ignored)
  prompts/               V0–V6, judge, verifier YAML files
  src/policypilot/
    __init__.py
    models.py            RunRecord
    pipeline.py          run_query() — main entry point
    ingest/              Doc2Dial loading + chunking
    retrieval/           FAISS index, retrieve()
    llm/                 client.py — all API calls
    prompting/           loader.py — YAML prompt loading
    tools/               calculator, search_documents, get_section
    agent/               react.py, verifier.py
    router/              route.py, train.py
    eval/                harness, metrics, stats
    app/                 Streamlit UI (4 tabs)
  eval/                  Golden sets (read-only after D1)
  router/data/           Router training data (read-only after D1)
  results/               All experiment outputs
    runs/                RunRecord JSONL files
    metrics/             Per-run metric CSVs
    param_sweep/         Sweep outputs
    figures/             matplotlib PNGs
    router/              Router evaluation outputs
    summary.csv          Aggregated metrics across all versions
  scripts/
    data/                download_all.py, inspect_schemas.py, build_eval_sets.py
    run_eval.py          Development evaluation (--split dev)
    run_final_eval.py    Test evaluation (once, on frozen prompts)
    run_sweep.py         Parameter and retrieval sweeps
    router_cv.py         25-run cross-validation
    router_final.py      Held-out test evaluation (once)
    token_budget.py      Token count table per prompt version
    autocot_build.py     AutoCoT demo generation
    make_report_assets.py Report tables and figures
    estimate_cost.py     Pre-run cost estimation
  tests/                 pytest suite
  docs/                  architecture.md, autocot.md, iteration_log.md
  report/assets/         LaTeX/Markdown tables and PNG figures
  slides/
```
