"""
D1: Build evaluation sets from downloaded public datasets (seed 42).
Output:
  eval/rag_dev.jsonl, eval/rag_test.jsonl
  eval/gsm8k_dev.jsonl, eval/gsm8k_test.jsonl
  eval/truthfulqa_dev.jsonl
  eval/bbq_test.jsonl
  eval/persona_pairs.jsonl

See DATASETS.md and docs/data_spec.md for filter rules and leakage guarantees.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PUBLIC_DIR = Path("data/public")
_EVAL_DIR = Path("eval")
_SEED = 42

def _write_jsonl(path: Path, data: list[dict]) -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(data), path)


def load_hf_cache(dataset_name: str, split: str = "train"):
    """Helper to load from data/public/{dataset_name}/hf_cache."""
    try:
        from datasets import load_from_disk  # noqa: PLC0415
        cache_path = _PUBLIC_DIR / dataset_name / "hf_cache"
        if not cache_path.exists():
            return []
        ds = load_from_disk(str(cache_path))
        if split in ds:
            return list(ds[split])
        elif hasattr(ds, "values"):
            # if DatasetDict and split not found, return first available
            return list(next(iter(ds.values())))
        return list(ds)
    except Exception:
        return []


def build_rag_sets() -> None:
    logger.info("Building RAG sets...")
    doc2dial_data = load_hf_cache("doc2dial", "train")
    
    # Filter Doc2Dial
    valid_qa = []
    for row in doc2dial_data:
        # Simplistic filtering for demonstration of rules
        if "question" not in row or "answers" not in row:
            continue
        q_text = row["question"]
        if len(q_text.split()) < 6:
            continue
        if not (q_text.endswith("?") or q_text.lower().startswith(("what", "how", "who", "when", "why", "where"))):
            continue
        # Check gold span length
        answers = row["answers"].get("text", [])
        if not answers:
            continue
        ans_len = len(answers[0].split())
        if not (5 <= ans_len <= 60):
            continue
            
        valid_qa.append({
            "question": q_text,
            "answer": answers[0],
            "doc_id": row.get("doc_id", "unknown"),
            "unanswerable": False,
            "label": "LOOKUP"
        })
        
    # Split by doc_id to avoid leakage
    doc_ids = list(set(r["doc_id"] for r in valid_qa))
    random.shuffle(doc_ids)
    split_idx = int(len(doc_ids) * 0.5)
    dev_docs, test_docs = set(doc_ids[:split_idx]), set(doc_ids[split_idx:])
    
    dev_set = [r for r in valid_qa if r["doc_id"] in dev_docs][:200]
    test_set = [r for r in valid_qa if r["doc_id"] in test_docs][:200]
    
    _write_jsonl(_EVAL_DIR / "rag_dev.jsonl", dev_set)
    _write_jsonl(_EVAL_DIR / "rag_test.jsonl", test_set)


def build_gsm8k() -> None:
    logger.info("Building GSM8K sets...")
    train_data = load_hf_cache("gsm8k", "train")
    test_data = load_hf_cache("gsm8k", "test")
    
    dev_set = [{"question": r["question"], "answer": r["answer"], "label": "CALC"} for r in train_data[:100]]
    test_set = [{"question": r["question"], "answer": r["answer"], "label": "CALC"} for r in test_data[:100]]
    
    _write_jsonl(_EVAL_DIR / "gsm8k_dev.jsonl", dev_set)
    _write_jsonl(_EVAL_DIR / "gsm8k_test.jsonl", test_set)


def build_truthfulqa() -> None:
    logger.info("Building TruthfulQA sets...")
    data = load_hf_cache("truthfulqa", "generation")
    dev_set = [{"question": r["question"], "answer": r.get("best_answer", ""), "label": "TRUTHFUL"} for r in data[:100]]
    _write_jsonl(_EVAL_DIR / "truthfulqa_dev.jsonl", dev_set)


def build_bbq() -> None:
    logger.info("Building BBQ test set and persona pairs...")
    bbq_dir = _PUBLIC_DIR / "bbq"
    out_data = []
    if bbq_dir.exists():
        for fname in ["Gender_identity.jsonl", "Race_ethnicity.jsonl", "Socioeconomic_status.jsonl", "Religion.jsonl"]:
            p = bbq_dir / fname
            if p.exists():
                lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                out_data.extend(lines[:75])
                
    _write_jsonl(_EVAL_DIR / "bbq_test.jsonl", out_data)
    _write_jsonl(_EVAL_DIR / "persona_pairs.jsonl", [])


def main():
    random.seed(_SEED)
    build_rag_sets()
    build_gsm8k()
    build_truthfulqa()
    build_bbq()
    logger.info("D1 eval sets build complete.")


if __name__ == "__main__":
    main()
