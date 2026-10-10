"""
RunRecord — the atomic unit of every experiment.

Every metric in results/ is computed from RunRecord objects.
"Every number in the report must come from a file in results/" (AGENTS.md).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RunRecord:
    """One Q&A exchange: everything needed to compute any metric."""

    question: str
    prompt_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model: str = ""
    temperature: float = 0.0
    top_p: float = 1.0

    # Retrieval
    retrieved_section_ids: list[str] = field(default_factory=list)

    # Raw and parsed outputs
    raw_output: str = ""
    parsed_output: dict[str, Any] | None = None
    parse_error: str | None = None  # Set if JSON parsing failed after repair

    # ReAct trace (V5, V6)
    react_trace: list[dict[str, Any]] = field(default_factory=list)

    # Verifier (V6)
    verifier_result: list[dict[str, Any]] | None = None

    # Router
    router_label: str | None = None
    router_confidence: float | None = None

    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None
    cached: bool = False

    # Evaluation (filled by eval harness)
    judge_score: float | None = None
    judge_reason: str | None = None
    hallucination_flag: bool | None = None  # True if any claim unsupported
    citation_accurate: bool | None = None
    refused: bool | None = None
    gold_in_top_k: bool | None = None
    arithmetic_exact: bool | None = None
    format_valid: bool | None = None

    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    prompt_file_hash: str = ""   # SHA-256 of the prompt YAML (set by pipeline)

    def model_dump(self) -> dict:
        return asdict(self)

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)
