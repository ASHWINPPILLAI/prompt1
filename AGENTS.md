# PolicyPilot: agent instructions

## Goal
RAG + ReAct question-answering assistant over public government-service documents (Doc2Dial: SSA, VA, DMV),
built as a Prompt Engineering course project. The deliverable is judged on prompt design, parameter tuning,
a working demo, rigorous evaluation, and a report. Clarity and reproducibility beat cleverness.

## Hard rules
- Python 3.11, type hints, small functions, docstrings that say WHY.
- Never hard-code API keys. Read from .env (python-dotenv). Keep .env out of git.
- Never fabricate results, metrics, citations or dataset rows. Every number in docs or report must come from
  a file under results/.
- Never edit eval/*.jsonl, router/data/router.jsonl, or data/public/**. They are human-authored or
  downloaded ground truth. Read-only during phases after D1.
- Every dataset comes from a public source listed in DATASETS.md. Never write, edit, paraphrase or generate
  dataset rows by hand or with an LLM.
- data/public/ and eval/*.jsonl are produced only by scripts/data/*. Otherwise they are read-only.
- Record every dataset in DATASETS.md: name, URL, version or commit, licence, download date, row counts
  after filtering, citation.
- If a download or loader fails (network blocked, script-based dataset refused), stop and report the exact
  error. Never substitute synthetic data.
- Split with seed 42. Split Doc2Dial by document id and dialogue id. Never let one document or dialogue
  appear in two splits.
- Test splits may only be used by scripts/run_final_eval.py and scripts/router_final.py.
- Prompts live ONLY in prompts/*.yaml. No prompt text inside Python except loading code.
- All LLM calls go through src/policypilot/llm/client.py (retry, timeout, cache, usage log). No direct
  SDK calls elsewhere.
- Every experiment writes raw outputs to results/runs/<run_id>.jsonl and a config snapshot beside it.
- Test set (split=test) may only be used by scripts/run_final_eval.py. Development uses split=dev.
- Prefer the standard library plus: openai, pydantic, pyyaml, tiktoken, sentence-transformers, faiss-cpu,
  pymupdf, pandas, matplotlib, streamlit, pytest, tenacity, scikit-learn, ruff, datasets, scipy.
- When unsure, ask or leave a TODO with a note. Do not invent behaviour.

## Commands
- make setup       — pip install -r requirements.txt
- make check       — ruff check . && pytest -q
- make data        — download_all.py + build_eval_sets.py + build_router_data.py
- make ingest      — python -m policypilot.ingest
- make index       — python -m policypilot.retrieval.build
- make eval-dev    — python scripts/run_eval.py --split dev --versions all
- make eval-test   — python scripts/run_final_eval.py
- make sweep       — python scripts/run_sweep.py
- make router-cv   — python scripts/router_cv.py
- make router-final — python scripts/router_final.py
- make report      — python scripts/make_report_assets.py
- make app         — streamlit run src/policypilot/app/main.py

## Definition of done for any task
Code runs, tests added or updated, `make check` passes, README or docs updated, a short summary of
what changed and how to verify it.

## Style
Plain, readable code. No unnecessary abstractions. Log with the logging module, not print.
