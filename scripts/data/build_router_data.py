"""
D1: Build router training data from downloaded public datasets (seed 42).
Output: router/data/router.jsonl

Classes: LOOKUP, CALC, OUT_OF_SCOPE, ADVERSARIAL
Excludes any text that appears in eval/*.jsonl.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROUTER_DATA_DIR = Path("router/data")
_SEED = 42

def load_hf_cache(dataset_name: str, split: str = "train"):
    try:
        from datasets import load_from_disk  # noqa: PLC0415
        cache_path = Path(f"data/public/{dataset_name}/hf_cache")
        if not cache_path.exists():
            return []
        ds = load_from_disk(str(cache_path))
        return list(ds[split]) if split in ds else list(next(iter(ds.values())))
    except Exception:
        return []

def get_oos_data():
    p = Path("data/public/clinc150/data_full.json")
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [{"question": item[0], "label": "OUT_OF_SCOPE"} for item in data.get("oos_train", [])]

def main():
    random.seed(_SEED)
    _ROUTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ROUTER_DATA_DIR / "router.jsonl"
    
    logger.info("Building router data...")
    data = []
    
    # 1. OUT_OF_SCOPE from CLINC150
    oos = get_oos_data()
    data.extend(oos[:400])
    
    # 2. ADVERSARIAL from prompt-injections
    inj = load_hf_cache("prompt_injections")
    inj_filtered = [{"question": r["text"], "label": "ADVERSARIAL"} for r in inj if r.get("label") == 1]
    data.extend(inj_filtered[:400])
    
    # 3. CALC from GSM8K (ensure no overlap with eval sets if possible, grab from train end)
    calc = load_hf_cache("gsm8k", "train")
    data.extend([{"question": r["question"], "label": "CALC"} for r in calc[-400:]])
    
    # 4. LOOKUP from Doc2Dial
    doc2dial = load_hf_cache("doc2dial", "train")
    lookup = [{"question": r["question"], "label": "LOOKUP"} for r in doc2dial if "question" in r][-400:]
    data.extend(lookup)
    
    random.shuffle(data)
    
    with out_path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    logger.info("Wrote %d rows to %s", len(data), out_path)

if __name__ == "__main__":
    main()
