"""
Main pipeline — run_query() is the single entry point for all prompt versions.

One pipeline handles V0 to V6 and 'routed'; all differences come from the
prompt YAML, the retrieval flag, the ReAct flag, and the verifier flag.

RunRecord is the unit every metric is computed from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from policypilot.agent.react import ReActAgent
from policypilot.agent.verifier import run_verifier
from policypilot.llm.client import LLMClient, get_client
from policypilot.models import RunRecord
from policypilot.prompting.loader import build_messages, load_prompt

logger = logging.getLogger(__name__)

_RUNS_DIR = Path("results/runs")
_REACT_PROMPT_IDS = {"v5", "v6", "routed"}  # prompt ids that use the ReAct loop
_VERIFIER_PROMPT_IDS = {"v6"}               # prompt ids that also run the verifier

_REFUSAL_TEMPLATE = (
    "The provided documents do not cover this question. "
    "Please check the official agency website or contact the relevant office."
)
_ADVERSARIAL_TEMPLATE = (
    "This question appears to contain an instruction injection. "
    "I only answer questions about public government service documents."
)


def _hash_prompt_file(prompt_id: str) -> str:
    """SHA-256 of the prompt YAML file for traceability."""
    path = Path("prompts") / f"{prompt_id}.yaml"
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _format_chunks_for_prompt(chunks: list[dict]) -> tuple[str, str]:
    """Return (plain_text, text_with_ids) for template rendering."""
    plain = "\n---\n".join(c["text"] for c in chunks)
    with_ids = "\n---\n".join(
        f"[{c['clause_id']}]\n{c['text']}" for c in chunks
    )
    return plain, with_ids


def stream_query(
    question: str,
    prompt_id: str,
    history: list[dict[str, str]] | None = None
) -> Iterator[str]:
    # If the user selects a ReAct/tool-calling agent, native streaming breaks the loop.
    # We run it synchronously and simulate a stream for the final answer.
    if "react" in prompt_id:
        record = run_query(question, prompt_id, history=history)
        yield "*(Agent used internal tools to compute this)*\n\n"
        answer = record.parsed_output.get("answer", record.raw_output) if record.parsed_output else record.raw_output
        for word in str(answer).split():
            yield word + " "
            time.sleep(0.01)
        return

    prompt = load_prompt(prompt_id)
    messages = build_messages(prompt, question, history=history)
    client = get_client()
    yield from client.stream_chat(messages, model=client.model)


def run_query(
    question: str,
    prompt_id: str,
    config: dict[str, Any] | None = None,
    *,
    client: LLMClient | None = None,
    top_k: int = 5,
    run_id: str | None = None,
    history: list[dict[str, str]] | None = None
) -> RunRecord:
    """
    Run a single question through the specified prompt version pipeline.

    Args:
        question:   The user's question string.
        prompt_id:  e.g. "v0_baseline", "v5_react", "routed"
        config:     Optional override dict (temperature, top_p, max_tokens).
        client:     LLM client; uses the shared default if None.
        top_k:      Number of chunks to retrieve.
        run_id:     Optional ID for logging; auto-generated if None.

    Returns a RunRecord containing everything needed to compute any metric.
    """
    import uuid
    rid = run_id or str(uuid.uuid4())[:8]
    cfg = config or {}
    temperature = cfg.get("temperature", 0.2)
    top_p = cfg.get("top_p", 1.0)
    max_tokens = cfg.get("max_tokens", 600)

    if client is None:
        client = get_client()

    record = RunRecord(
        run_id=rid,
        question=question,
        prompt_id=prompt_id,
        model=client.model,
        temperature=temperature,
        top_p=top_p,
        prompt_file_hash=_hash_prompt_file(prompt_id),
    )

    # ── Router intercept ──────────────────────────────────────────────────────
    if prompt_id == "routed":
        return _run_routed(question, config=cfg, client=client, run_id=rid)

    # ── Load prompt ───────────────────────────────────────────────────────────
    try:
        prompt = load_prompt(prompt_id)
    except FileNotFoundError:
        record.raw_output = f"ERROR: prompt '{prompt_id}' not found"
        return record

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks: list[dict] = []
    if prompt.needs_retrieval:
        try:
            from policypilot.retrieval import retrieve  # noqa: PLC0415
            chunks = retrieve(question, k=top_k)
            record.retrieved_section_ids = [c["clause_id"] for c in chunks]
        except FileNotFoundError as e:
            logger.warning("Retrieval failed: %s", e)

    plain_chunks, chunks_with_ids = _format_chunks_for_prompt(chunks)

    # ── ReAct branch (V5, V6) ─────────────────────────────────────────────────
    if prompt_id in _REACT_PROMPT_IDS:
        messages = build_messages(
            prompt,
            question=question,
            retrieved_chunks=plain_chunks,
            retrieved_chunks_with_ids=chunks_with_ids,
            history=history
        )
        agent = ReActAgent(client=client)
        result = agent.run(messages, run_id=rid, prompt_id=prompt_id)

        record.raw_output = result["final_answer"]
        record.react_trace = result["react_trace"]
        record.prompt_tokens = result["total_tokens"]["prompt"]
        record.completion_tokens = result["total_tokens"]["completion"]
        record.latency_s = result["latency_s"]

        answer_text = result["final_answer"]

        # Verifier pass (V6 only)
        if prompt_id in _VERIFIER_PROMPT_IDS and chunks:
            ver = run_verifier(answer_text, chunks_with_ids, run_id=rid)
            record.verifier_result = ver["claims"]
            record.hallucination_flag = ver["hallucination"]
            answer_text = ver["verified_answer"]
            record.prompt_tokens += ver["usage"]["prompt_tokens"]
            record.completion_tokens += ver["usage"]["completion_tokens"]

        # Try parse as JSON
        record.parsed_output, record.parse_error = _parse_json_output(answer_text)
        return record

    # ── Standard (V0–V4b) ────────────────────────────────────────────────────
    messages = build_messages(
        prompt,
        question=question,
        retrieved_chunks=plain_chunks,
        retrieved_chunks_with_ids=chunks_with_ids,
        history=history
    )
    t0 = time.perf_counter()
    response = client.chat(
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        run_id=rid,
        prompt_id=prompt_id,
    )
    record.latency_s = time.perf_counter() - t0
    record.raw_output = response["content"]
    record.prompt_tokens = response["usage"]["prompt_tokens"]
    record.completion_tokens = response["usage"]["completion_tokens"]
    record.cached = response["cached"]

    record.parsed_output, record.parse_error = _parse_json_output(response["content"])
    return record


def _parse_json_output(text: str) -> tuple[dict | None, str | None]:
    """
    Try to parse the model output as JSON.

    First attempt: raw parse.
    Second attempt (repair): extract the first {...} block and try again.

    WHY repair? Models occasionally wrap JSON in markdown code fences
    or add a brief preamble sentence.
    """
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Repair: find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end]), None
        except json.JSONDecodeError as e:
            return None, str(e)

    return None, "No JSON object found in output"


def _run_routed(
    question: str,
    config: dict | None = None,
    *,
    client: LLMClient,
    run_id: str,
) -> RunRecord:
    """Route a question to the appropriate prompt version via the trained router."""
    try:
        from policypilot.router.route import predict  # noqa: PLC0415
        label, confidence = predict(question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Router failed (%s); falling back to v6", exc)
        label, confidence = "FALLBACK", 0.0

    route_map = {
        "LOOKUP": "v4_cot",
        "CALC": "v5_react",
        "OUT_OF_SCOPE": None,
        "ADVERSARIAL": None,
        "FALLBACK": "v6_verifier",
    }
    target_prompt = route_map.get(label, "v6_verifier")

    # OUT_OF_SCOPE and ADVERSARIAL: fixed refusal, no LLM call
    if target_prompt is None:
        text = _ADVERSARIAL_TEMPLATE if label == "ADVERSARIAL" else _REFUSAL_TEMPLATE
        record = RunRecord(
            run_id=run_id,
            question=question,
            prompt_id="routed",
            router_label=label,
            router_confidence=confidence,
            raw_output=text,
            refused=True,
        )
        return record

    record = run_query(question, target_prompt, config, client=client, run_id=run_id)
    record.prompt_id = "routed"
    record.router_label = label
    record.router_confidence = confidence
    return record


def save_run(record: RunRecord, run_dir: Path = _RUNS_DIR) -> Path:
    """Append a RunRecord to results/runs/<prompt_id>.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{record.prompt_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return path
