"""
PolicyPilot package root.

Sub-packages:
  ingest     — Doc2Dial document loading and chunking
  retrieval  — FAISS embedding index and query interface
  llm        — OpenAI-compatible LLM client (retry, cache, usage log)
  prompting  — YAML prompt loader and template renderer
  tools      — Calculator and document search tools
  agent      — ReAct loop and verifier
  router     — Question router (4-class classifier)
  eval       — Evaluation harness and metrics
  app        — Streamlit UI
"""
