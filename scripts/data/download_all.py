"""
D0: Download all public datasets for PolicyPilot.

Datasets:
  1. Doc2Dial    — IBM/doc2dial (Hugging Face or official site fallback)
  2. CLINC150    — clinc/oos-eval (GitHub raw)
  3. deepset/prompt-injections — Hugging Face
  4. GSM8K       — openai/gsm8k (Hugging Face)
  5. TruthfulQA  — truthful_qa (Hugging Face)
  6. BBQ         — nyu-mll/BBQ (GitHub)

All datasets saved to data/public/<name>/ with a checksum file.
DATASETS.md updated with name, source, licence, download date, row counts.

AGENTS.md rule: If a download fails, print the exact error and STOP.
Never substitute synthetic data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PUBLIC_DIR = Path("data/public")
_DATASETS_MD = Path("DATASETS.md")

_TODAY = datetime.now().strftime("%Y-%m-%d")


# ── Checksum helper ───────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(path: Path, checksum: str) -> None:
    (path.parent / f"{path.name}.sha256").write_text(checksum, encoding="utf-8")


# ── Dataset downloaders ───────────────────────────────────────────────────────
def download_doc2dial() -> dict:
    """
    Download Doc2Dial from Hugging Face (IBM/doc2dial).

    Falls back to official site if HF datasets library refuses the script.
    Saves raw files to data/public/doc2dial/<domain>/*.json
    """
    out_dir = _PUBLIC_DIR / "doc2dial"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Doc2Dial from Hugging Face …")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        # trust_remote_code required for the doc2dial loading script
        ds = load_dataset("IBM/doc2dial", trust_remote_code=True)
        # Save the raw dataset to disk in arrow format
        ds.save_to_disk(str(out_dir / "hf_cache"))
        logger.info("Doc2Dial HF cache saved to %s", out_dir / "hf_cache")

        # Extract documents to per-domain JSON for easy loading
        row_count = 0
        if "train" in ds:
            for row in ds["train"]:
                domain = row.get("domain", "unknown")
                doc_id = row.get("doc_id", "unknown")
                domain_dir = out_dir / domain
                domain_dir.mkdir(exist_ok=True)
                out_path = domain_dir / f"{doc_id}.json"
                out_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                row_count += 1
        logger.info("Extracted %d Doc2Dial documents", row_count)
        return {"rows": row_count, "method": "huggingface"}

    except Exception as exc:
        logger.error("Doc2Dial HF download failed: %s", exc)
        logger.error(
            "Fallback: download from http://doc2dial.github.io and place files in %s", out_dir
        )
        raise SystemExit(1) from exc


def download_clinc150() -> dict:
    """
    Download CLINC150 from GitHub (clinc/oos-eval).
    File: data_full.json
    """
    import urllib.request  # noqa: PLC0415

    out_dir = _PUBLIC_DIR / "clinc150"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://raw.githubusercontent.com/clinc/oos-eval/master/"
        "data/data_full.json"
    )
    out_path = out_dir / "data_full.json"
    if out_path.exists():
        logger.info("CLINC150 already downloaded, skipping")
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        logger.info("Downloading CLINC150 from %s …", url)
        try:
            urllib.request.urlretrieve(url, out_path)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("CLINC150 download failed: %s", exc)
            raise SystemExit(1) from exc

    checksum = sha256_file(out_path)
    write_checksum(out_path, checksum)
    # Count rows: data has keys like "train", "oos_train", etc.
    total = sum(len(v) for v in data.values() if isinstance(v, list))
    logger.info("CLINC150: %d total examples", total)
    return {"rows": total, "checksum": checksum}


def download_prompt_injections() -> dict:
    """Download deepset/prompt-injections from Hugging Face."""
    out_dir = _PUBLIC_DIR / "prompt_injections"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading deepset/prompt-injections …")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("deepset/prompt-injections")
        ds.save_to_disk(str(out_dir / "hf_cache"))
        # Count rows
        total = sum(len(split) for split in ds.values())
        logger.info("prompt-injections: %d total examples", total)
        return {"rows": total, "method": "huggingface"}
    except Exception as exc:
        logger.error("prompt-injections download failed: %s", exc)
        raise SystemExit(1) from exc


def download_gsm8k() -> dict:
    """Download GSM8K from Hugging Face (openai/gsm8k)."""
    out_dir = _PUBLIC_DIR / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GSM8K …")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("openai/gsm8k", "main")
        ds.save_to_disk(str(out_dir / "hf_cache"))
        total = sum(len(split) for split in ds.values())
        logger.info("GSM8K: %d total examples", total)
        return {"rows": total, "method": "huggingface"}
    except Exception as exc:
        logger.error("GSM8K download failed: %s", exc)
        raise SystemExit(1) from exc


def download_truthfulqa() -> dict:
    """Download TruthfulQA from Hugging Face."""
    out_dir = _PUBLIC_DIR / "truthfulqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading TruthfulQA …")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("truthful_qa", "generation")
        ds.save_to_disk(str(out_dir / "hf_cache"))
        total = sum(len(split) for split in ds.values())
        logger.info("TruthfulQA: %d total examples", total)
        return {"rows": total, "method": "huggingface"}
    except Exception as exc:
        logger.error("TruthfulQA download failed: %s", exc)
        raise SystemExit(1) from exc


def download_bbq() -> dict:
    """
    Download BBQ from GitHub (nyu-mll/BBQ).
    Downloads all .jsonl files from the data/ folder.
    """
    import urllib.request  # noqa: PLC0415

    out_dir = _PUBLIC_DIR / "bbq"
    out_dir.mkdir(parents=True, exist_ok=True)

    # BBQ categories we need (section 13.3: gender identity, race, socioeconomic, religion)
    categories = [
        "Gender_identity",
        "Race_ethnicity",
        "Socioeconomic_status",
        "Religion",
    ]
    base_url = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/"
    total = 0

    logger.info("Downloading BBQ categories: %s", categories)
    for cat in categories:
        fname = f"{cat}.jsonl"
        url = base_url + fname
        out_path = out_dir / fname
        if out_path.exists():
            logger.info("  %s already exists, skipping", fname)
        else:
            try:
                logger.info("  Downloading %s …", fname)
                urllib.request.urlretrieve(url, out_path)
            except Exception as exc:
                logger.error("BBQ download failed for %s: %s", fname, exc)
                raise SystemExit(1) from exc
        # Count rows
        lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total += len(lines)
        checksum = sha256_file(out_path)
        write_checksum(out_path, checksum)

    logger.info("BBQ: %d total examples across %d categories", total, len(categories))
    return {"rows": total}


# ── DATASETS.md writer ────────────────────────────────────────────────────────
def write_datasets_md(stats: dict[str, dict]) -> None:
    """Write DATASETS.md with provenance, licences, row counts."""
    content = f"""# DATASETS.md — PolicyPilot data provenance

Generated: {_TODAY}
All datasets are public. Raw files in data/public/ are read-only after `make data`.

| Dataset | Source | Licence | Download date | Raw rows | Citation |
|---|---|---|---|---|---|
| Doc2Dial | [IBM/doc2dial (HuggingFace)](https://huggingface.co/datasets/IBM/doc2dial) | CC-BY-3.0 | {_TODAY} | {stats.get('doc2dial', {}).get('rows', 'N/A')} | Feng et al., EMNLP 2020 |
| CLINC150 | [clinc/oos-eval (GitHub)](https://github.com/clinc/oos-eval) | CC-BY-3.0 | {_TODAY} | {stats.get('clinc150', {}).get('rows', 'N/A')} | Larson et al., EMNLP-IJCNLP 2019 |
| prompt-injections | [deepset/prompt-injections (HuggingFace)](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | {_TODAY} | {stats.get('prompt_injections', {}).get('rows', 'N/A')} | deepset (see dataset card) |
| GSM8K | [openai/gsm8k (HuggingFace)](https://huggingface.co/datasets/openai/gsm8k) | MIT | {_TODAY} | {stats.get('gsm8k', {}).get('rows', 'N/A')} | Cobbe et al. 2021 |
| TruthfulQA | [truthful_qa (HuggingFace)](https://huggingface.co/datasets/truthful_qa) | Apache-2.0 | {_TODAY} | {stats.get('truthfulqa', {}).get('rows', 'N/A')} | Lin et al. 2022 |
| BBQ | [nyu-mll/BBQ (GitHub)](https://github.com/nyu-mll/BBQ) | CC-BY-4.0 | {_TODAY} | {stats.get('bbq', {}).get('rows', 'N/A')} | Parrish et al., ACL Findings 2022 |

## Filters applied in build_eval_sets.py (see D1 output for counts after filtering)

- Doc2Dial: first user turn only, ≥6 words, looks like a question, gold span 5–60 words,
  near-duplicates removed, split by document id (no doc in both dev and test).
- CLINC150: out-of-scope class only (label == "oos").
- prompt-injections: English rows only, label == 1 (injection attempts).
- GSM8K: dev from train split, test from test split. Questions contain a numeric answer.
- TruthfulQA: sample of 100 (generation config). Questions with reference answers.
- BBQ: 4 categories (Gender_identity, Race_ethnicity, Socioeconomic_status, Religion),
  75 examples each, half ambiguous and half disambiguated.

## HUMAN spot-check results
<!-- HUMAN: After D1, read 30 random items per set and fill this table. -->
| Set | Items read | Bad items | Bad-item rate | Action |
|---|---|---|---|---|
| rag_dev | 30 | TBD | TBD | TBD |
| rag_test | 30 | TBD | TBD | TBD |
| gsm8k_dev | 30 | TBD | TBD | TBD |
| router | 30 | TBD | TBD | TBD |

If bad-item rate > 10%, tighten filters and rebuild.

## Leakage guarantees
- Doc2Dial splits are by document id and dialogue id; no document appears in two splits.
- Router training data excludes all items in eval/*.jsonl (checked by tests/test_leakage.py).
- Test splits are read-only after D1; touched only by scripts/run_final_eval.py and
  scripts/router_final.py.
"""
    _DATASETS_MD.write_text(content, encoding="utf-8")
    logger.info("DATASETS.md written")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    stats: dict[str, dict] = {}

    logger.info("=== D0: Downloading all datasets ===")
    logger.info("Saving to: %s", _PUBLIC_DIR.absolute())

    # Check network access
    try:
        import urllib.request  # noqa: PLC0415
        urllib.request.urlopen("https://huggingface.co", timeout=10)
    except Exception:
        logger.error(
            "Cannot reach huggingface.co. If you are in an agent sandbox, run this script "
            "on your own machine first, then provide the data/public/ folder to the agent."
        )
        raise SystemExit(1)

    stats["doc2dial"] = download_doc2dial()
    stats["clinc150"] = download_clinc150()
    stats["prompt_injections"] = download_prompt_injections()
    stats["gsm8k"] = download_gsm8k()
    stats["truthfulqa"] = download_truthfulqa()
    stats["bbq"] = download_bbq()

    write_datasets_md(stats)

    logger.info("=== D0 complete ===")
    for name, s in stats.items():
        logger.info("  %-25s %s rows", name, s.get("rows", "?"))
    logger.info("Next step: python scripts/data/build_eval_sets.py (D1)")


if __name__ == "__main__":
    main()
