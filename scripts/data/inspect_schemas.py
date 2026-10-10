"""
D0.5: Schema inspection — read all downloaded datasets and produce
docs/doc2dial_schema.md with field schemas, example records, and counts.

Run BEFORE D1 (build_eval_sets.py) to avoid guessing field names.

AGENTS.md rule: Do not continue until the human has read this file.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PUBLIC_DIR = Path("data/public")
_DOCS_DIR = Path("docs")
_OUTPUT = _DOCS_DIR / "doc2dial_schema.md"


def load_jsonl_sample(path: Path, n: int = 3) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()][:n]


def inspect_doc2dial() -> str:
    """Inspect Doc2Dial HF cache or per-domain JSON files."""
    sections = []
    hf_cache = _PUBLIC_DIR / "doc2dial" / "hf_cache"

    if hf_cache.exists():
        try:
            from datasets import load_from_disk  # noqa: PLC0415
            ds = load_from_disk(str(hf_cache))
            sections.append("## Doc2Dial — Loaded from HF cache\n")
            sections.append(f"Available splits: {list(ds.keys())}\n")
            for split in ds:
                s = ds[split]
                sections.append(f"\n### Split: {split} ({len(s)} rows)\n")
                sections.append(f"Features: {s.features}\n")
                sections.append("\n**3 example rows:**\n```json\n")
                for i in range(min(3, len(s))):
                    sections.append(json.dumps(dict(s[i]), ensure_ascii=False, default=str)[:800])
                    sections.append("\n")
                sections.append("```\n")

                # Domain counts
                if "domain" in s.features:
                    from collections import Counter  # noqa: PLC0415
                    domains = Counter(s["domain"])
                    sections.append(f"\nDomain counts: {dict(domains)}\n")

                # Unanswerable marker check
                sections.append("\n**Unanswerable marker search:**\n")
                for field in s.features:
                    if any(kw in field.lower() for kw in ["unansw", "irrelevant", "cannot", "no_answer"]):
                        sections.append(f"  Found potential unanswerable field: `{field}`\n")
            return "".join(sections)
        except Exception as exc:
            sections.append(f"HF load failed ({exc}), inspecting per-domain JSON files.\n")

    # Fallback: inspect per-domain JSON files
    doc2dial_dir = _PUBLIC_DIR / "doc2dial"
    if not doc2dial_dir.exists():
        return "## Doc2Dial — NOT DOWNLOADED\nRun scripts/data/download_all.py first.\n"

    sections.append("## Doc2Dial — Per-domain JSON files\n")
    for domain_dir in sorted(doc2dial_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        files = list(domain_dir.glob("*.json"))
        sections.append(f"\n### Domain: {domain_dir.name} ({len(files)} documents)\n")
        if files:
            sample = json.loads(files[0].read_text(encoding="utf-8"))
            sections.append(f"Top-level keys: {list(sample.keys())}\n")
            # Show spans structure
            if "spans" in sample:
                sections.append(f"\nSpans count in first doc: {len(sample['spans'])}\n")
                if sample["spans"]:
                    sections.append(f"Span keys: {list(sample['spans'][0].keys())}\n")
                    sections.append("**3 sample spans:**\n```json\n")
                    for sp in sample["spans"][:3]:
                        sections.append(json.dumps(sp, ensure_ascii=False)[:300] + "\n")
                    sections.append("```\n")
            sections.append("```json\n" + json.dumps({k: v for k, v in list(sample.items())[:5]}, ensure_ascii=False, default=str)[:600] + "\n```\n")
    return "".join(sections)


def inspect_clinc150() -> str:
    path = _PUBLIC_DIR / "clinc150" / "data_full.json"
    if not path.exists():
        return "## CLINC150 — NOT DOWNLOADED\n"
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## CLINC150\n", f"Top-level keys: {list(data.keys())}\n"]
    for key in list(data.keys())[:4]:
        val = data[key]
        if isinstance(val, list) and val:
            lines.append(f"\n`{key}`: {len(val)} items. First 2:\n")
            for item in val[:2]:
                lines.append(f"  {item}\n")
    # OOS count
    if "oos_train" in data:
        lines.append(f"\nOut-of-scope (oos_train) count: {len(data['oos_train'])}\n")
    if "oos_test" in data:
        lines.append(f"Out-of-scope (oos_test) count: {len(data['oos_test'])}\n")
    return "".join(lines)


def inspect_prompt_injections() -> str:
    hf_cache = _PUBLIC_DIR / "prompt_injections" / "hf_cache"
    if not hf_cache.exists():
        return "## prompt-injections — NOT DOWNLOADED\n"
    try:
        from datasets import load_from_disk  # noqa: PLC0415
        ds = load_from_disk(str(hf_cache))
        lines = ["## deepset/prompt-injections\n", f"Splits: {list(ds.keys())}\n"]
        for split in ds:
            s = ds[split]
            lines.append(f"\n### {split}: {len(s)} rows\nFeatures: {s.features}\n")
            if len(s) > 0:
                lines.append("**3 examples:**\n```json\n")
                for i in range(min(3, len(s))):
                    lines.append(json.dumps(dict(s[i]), ensure_ascii=False)[:400] + "\n")
                lines.append("```\n")
            # Count by label
            if "label" in s.features:
                from collections import Counter  # noqa: PLC0415
                lines.append(f"Label distribution: {dict(Counter(s['label']))}\n")
        return "".join(lines)
    except Exception as exc:
        return f"## prompt-injections — load failed: {exc}\n"


def inspect_gsm8k() -> str:
    hf_cache = _PUBLIC_DIR / "gsm8k" / "hf_cache"
    if not hf_cache.exists():
        return "## GSM8K — NOT DOWNLOADED\n"
    try:
        from datasets import load_from_disk  # noqa: PLC0415
        ds = load_from_disk(str(hf_cache))
        lines = ["## GSM8K\n", f"Splits: {list(ds.keys())}\n"]
        for split in ds:
            s = ds[split]
            lines.append(f"\n### {split}: {len(s)} rows\nFeatures: {s.features}\n")
            if len(s) > 0:
                lines.append("**2 examples:**\n```json\n")
                for i in range(min(2, len(s))):
                    lines.append(json.dumps(dict(s[i]), ensure_ascii=False)[:600] + "\n")
                lines.append("```\n")
        return "".join(lines)
    except Exception as exc:
        return f"## GSM8K — load failed: {exc}\n"


def inspect_truthfulqa() -> str:
    hf_cache = _PUBLIC_DIR / "truthfulqa" / "hf_cache"
    if not hf_cache.exists():
        return "## TruthfulQA — NOT DOWNLOADED\n"
    try:
        from datasets import load_from_disk  # noqa: PLC0415
        ds = load_from_disk(str(hf_cache))
        lines = ["## TruthfulQA\n", f"Splits: {list(ds.keys())}\n"]
        for split in ds:
            s = ds[split]
            lines.append(f"\n### {split}: {len(s)} rows\nFeatures: {s.features}\n")
            if len(s) > 0:
                lines.append("**2 examples:**\n```json\n")
                for i in range(min(2, len(s))):
                    lines.append(json.dumps(dict(s[i]), ensure_ascii=False)[:600] + "\n")
                lines.append("```\n")
        return "".join(lines)
    except Exception as exc:
        return f"## TruthfulQA — load failed: {exc}\n"


def inspect_bbq() -> str:
    bbq_dir = _PUBLIC_DIR / "bbq"
    if not bbq_dir.exists():
        return "## BBQ — NOT DOWNLOADED\n"
    lines = ["## BBQ\n"]
    for jsonl_file in sorted(bbq_dir.glob("*.jsonl")):
        items = [json.loads(l) for l in jsonl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines.append(f"\n### {jsonl_file.name}: {len(items)} rows\n")
        if items:
            lines.append(f"Keys: {list(items[0].keys())}\n")
            lines.append("**2 examples:**\n```json\n")
            for item in items[:2]:
                lines.append(json.dumps(item, ensure_ascii=False)[:400] + "\n")
            lines.append("```\n")
    return "".join(lines)


def main() -> None:
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== D0.5: Schema inspection ===")

    sections = [
        "# docs/doc2dial_schema.md — Dataset Schema Inspection\n\n",
        "> **HUMAN GATE**: Read this file before running D1 (build_eval_sets.py).\n",
        "> Verify field names, confirm the unanswerable marker, and check span structure.\n\n",
        inspect_doc2dial(),
        "\n---\n\n",
        inspect_clinc150(),
        "\n---\n\n",
        inspect_prompt_injections(),
        "\n---\n\n",
        inspect_gsm8k(),
        "\n---\n\n",
        inspect_truthfulqa(),
        "\n---\n\n",
        inspect_bbq(),
    ]

    _OUTPUT.write_text("".join(sections), encoding="utf-8")
    logger.info("Schema inspection written to %s", _OUTPUT)
    logger.info("HUMAN: Read docs/doc2dial_schema.md before proceeding to D1.")


if __name__ == "__main__":
    main()
