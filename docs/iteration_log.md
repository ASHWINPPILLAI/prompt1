# docs/iteration_log.md — Prompt iteration log.
# HUMAN: Add 2–3 rows per prompt version after dev testing.
# This feeds directly into report Chapter 4 and proves the work is yours.

# Iteration Log

| Date | Version | Dev metric (before) | What failed | What I changed | Dev metric (after) | Notes |
|---|---|---|---|---|---|---|
| 2026-10-10 | V0 | — | Baseline: high hallucination, no citations | — (this is the floor) | Correctness: TBD%, Hallucination: TBD% | Run after D1 + index are ready |
| TBD | V1 | Correctness: V0 score | TBD failure observed | Added RAG context and persona | TBD | |
| TBD | V2 | Correctness: V1 score | TBD failure | Added rules, JSON schema, refusal | TBD | |
| TBD | V3 | Correctness: V2 score | TBD failure | Added 4 few-shot examples | TBD | |
| TBD | V4 | Correctness: V3 score | TBD failure | Added 5-step CoT rule | TBD | |
| TBD | V4b | Correctness: V4 score | TBD failure | Replaced hand-written demos with AutoCoT | TBD | |
| TBD | V5 | Correctness: V4b score | TBD failure | Added ReAct + calculator | TBD | |
| TBD | V6 | Correctness: V5 score | TBD failure | Added verifier pass | TBD | |

## How to fill this in

For each version:
1. Run `make eval-dev --versions v<N>` (dev split only, never test split)
2. Read through 5–10 answers manually
3. Note the most common failure mode (e.g., "cites wrong section", "invents a deadline")
4. Describe what you changed in the prompt and why
5. Record the dev metric after the change

The "What failed" and "What I changed" columns go into report Chapter 4, one
subsection per version. Without these notes the chapter reads like a manual.
