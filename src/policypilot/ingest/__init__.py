"""
Doc2Dial document ingestion.

Reads Doc2Dial documents from data/public/doc2dial/ and produces:
  data/processed/clauses.jsonl  — one record per document section/span
  data/processed/chunks.jsonl   — sections split to ≤ chunk_tokens with overlap

WHY sections rather than fixed-size pages?
Doc2Dial documents have explicit section structure (id_sp, tag, text_sp).
Preserving section boundaries means chunk_ids map 1:1 to citation section_ids,
so "cite clause R5.3" always refers to the same text regardless of chunk size.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PUBLIC_DIR = Path("data/public/doc2dial")
_PROCESSED_DIR = Path("data/processed")

# Domains to index (studentaid is held out as unanswerable pool)
INDEXED_DOMAINS = {"ssa", "va", "dmv"}


def _token_count(text: str) -> int:
    """Approximate token count: ~4 chars per token (tiktoken unavailable here)."""
    # WHY approximation? The ingest step doesn't call the LLM; we use tiktoken
    # in the retrieval and token-budget scripts where accuracy matters.
    return max(1, len(text) // 4)


def load_doc2dial_documents(raw_dir: Path = _PUBLIC_DIR) -> list[dict[str, Any]]:
    """
    Load Doc2Dial documents from the downloaded JSON files.

    Returns a list of dicts with keys:
      doc_id, domain, title, sections (list of {id_sp, tag, text_sp, start_sp, end_sp})
    """
    docs: list[dict[str, Any]] = []
    for domain_dir in sorted(raw_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if domain not in INDEXED_DOMAINS:
            logger.info("Skipping domain '%s' (held out or unknown)", domain)
            continue
        for doc_file in sorted(domain_dir.glob("*.json")):
            try:
                data = json.loads(doc_file.read_text(encoding="utf-8"))
                docs.append({**data, "domain": domain})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", doc_file, exc)
    logger.info("Loaded %d documents from domains: %s", len(docs), INDEXED_DOMAINS)
    return docs


def make_clauses(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert Doc2Dial span records into clause-level chunks.

    Each clause has: clause_id (doc_id#id_sp), doc_id, domain, section_id,
    title, text, token_count.
    """
    clauses: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.get("doc_id", "")
        domain = doc.get("domain", "")
        title = doc.get("title", "")
        for span in doc.get("spans", []):
            id_sp = span.get("id_sp", "")
            text = span.get("text_sp", "").strip()
            if not text:
                continue
            clauses.append(
                {
                    "clause_id": f"{doc_id}#{id_sp}",
                    "doc_id": doc_id,
                    "domain": domain,
                    "section_id": id_sp,
                    "title": title,
                    "text": text,
                    "token_count": _token_count(text),
                }
            )
    logger.info("Produced %d clause records", len(clauses))
    return clauses


def make_chunks(
    clauses: list[dict[str, Any]],
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """
    Split clauses longer than chunk_tokens into overlapping sub-chunks.

    Short clauses (≤ chunk_tokens) become a single chunk unchanged.
    The clause_id is preserved so citations always trace back to a real section.
    """
    chunks: list[dict[str, Any]] = []
    for clause in clauses:
        text = clause["text"]
        words = text.split()
        # Approximate words per token: 0.75 words/token → chunk_tokens * 0.75 words
        chunk_words = max(1, int(chunk_tokens * 0.75))
        overlap_words = max(0, int(overlap_tokens * 0.75))

        if len(words) <= chunk_words:
            chunks.append({**clause, "part": 0, "num_parts": 1})
            continue

        part = 0
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    **clause,
                    "text": chunk_text,
                    "token_count": _token_count(chunk_text),
                    "part": part,
                    "num_parts": -1,  # updated below
                }
            )
            if end == len(words):
                break
            start = end - overlap_words
            part += 1

        # Back-fill num_parts
        for c in chunks[-part - 1 :]:
            c["num_parts"] = part + 1

    logger.info("Produced %d chunks from %d clauses", len(chunks), len(clauses))
    return chunks


def run_ingest(
    raw_dir: Path = _PUBLIC_DIR,
    out_dir: Path = _PROCESSED_DIR,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
) -> None:
    """Entry point: ingest → clauses.jsonl + chunks.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_doc2dial_documents(raw_dir)
    if not docs:
        raise RuntimeError(
            f"No documents found in {raw_dir}. "
            "Run 'make data' (scripts/data/download_all.py) first."
        )

    clauses = make_clauses(docs)
    chunks = make_chunks(clauses, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    clauses_path = out_dir / "clauses.jsonl"
    chunks_path = out_dir / "chunks.jsonl"

    with clauses_path.open("w", encoding="utf-8") as f:
        for c in clauses:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Wrote %d clauses to %s", len(clauses), clauses_path)
    logger.info("Wrote %d chunks to %s", len(chunks), chunks_path)

    # Summary
    import statistics
    token_counts = [c["token_count"] for c in clauses]
    print(f"\n=== Ingestion Summary ===")
    print(f"Documents: {len(docs)}")
    print(f"Clauses:   {len(clauses)}")
    print(f"Chunks:    {len(chunks)}")
    if token_counts:
        print(f"Clause tokens — mean: {statistics.mean(token_counts):.0f}, "
              f"median: {statistics.median(token_counts):.0f}, "
              f"max: {max(token_counts)}")
    print(f"Output:    {out_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ingest()
