"""
Harness for running evaluations across prompt versions and datasets.

The harness iterates over an evaluation dataset (e.g., eval/rag_dev.jsonl),
calls run_query for each item, computes metrics, and writes RunRecords.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from policypilot.pipeline import run_query
from policypilot.eval.metrics import calculate_metrics

logger = logging.getLogger(__name__)

_EVAL_DIR = Path("eval")
_RUNS_DIR = Path("results/runs")


def load_dataset(filename: str) -> list[dict[str, Any]]:
    path = _EVAL_DIR / filename
    if not path.exists():
        logger.error(f"Dataset {filename} not found at {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(
    dataset_name: str,
    prompt_id: str,
    config: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Run evaluation for a specific prompt version on a dataset.
    
    Args:
        dataset_name: e.g., "rag_dev.jsonl"
        prompt_id: e.g., "v2_rules_json"
        config: Override config (temperature, etc.)
        limit: Max rows to run (for testing)
        
    Returns:
        (records, aggregate_metrics)
    """
    data = load_dataset(dataset_name)
    if limit:
        data = data[:limit]
        
    if not data:
        return [], {}

    logger.info(f"Evaluating {prompt_id} on {dataset_name} ({len(data)} items)")
    
    records = []
    
    for row in tqdm(data, desc=f"{prompt_id} on {dataset_name}"):
        question = row.get("question", row.get("text", ""))
        if not question:
            continue
            
        record = run_query(question, prompt_id, config=config)
        
        # Inject ground truth metadata into the record for metric calculation
        record_dict = record.model_dump()
        record_dict["_ground_truth"] = row
        records.append(record_dict)

        # Save record progressively
        _save_record(record_dict, prompt_id)
        
    metrics = calculate_metrics(records, dataset_name)
    logger.info(f"Metrics for {prompt_id}: {json.dumps(metrics, indent=2)}")
    
    return records, metrics


def _save_record(record_dict: dict, prompt_id: str) -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RUNS_DIR / f"{prompt_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        # Don't save the full _ground_truth in the runs file to save space, just standard record
        clean_record = {k: v for k, v in record_dict.items() if k != "_ground_truth"}
        f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

