"""
Build the FAISS index from data/processed/chunks.jsonl.

Run once: `make index` or `python -m policypilot.retrieval.build`

Steps:
1. Load all chunks from chunks.jsonl
2. Embed with MiniLM (batched for efficiency)
3. L2-normalise (so inner product = cosine similarity)
4. Build FAISS IndexFlatIP and write to data/index/
5. Write metadata (clause_id, text, etc.) to chunks_meta.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
_INDEX_DIR = Path("data/index")
_BATCH_SIZE = 256


def build_index(
    chunks_path: Path = _CHUNKS_PATH,
    index_dir: Path = _INDEX_DIR,
    batch_size: int = _BATCH_SIZE,
) -> None:
    """Embed all chunks and write the FAISS index + metadata."""
    import faiss  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {chunks_path}. Run 'make ingest' first."
        )

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Embedding %d chunks …", len(chunks))

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
        if (i // batch_size) % 10 == 0:
            logger.info("  Embedded %d / %d", min(i + batch_size, len(texts)), len(texts))

    matrix = np.vstack(all_vecs).astype("float32")
    dim = matrix.shape[1]
    logger.info("Vector matrix: shape=%s, dim=%d", matrix.shape, dim)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "chunks.index"))
    logger.info("FAISS index written: %d vectors", index.ntotal)

    # Write metadata (everything except large fields we don't need at query time)
    meta_path = index_dir / "chunks_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Metadata written: %s", meta_path)

    print(f"\n=== Index built: {index.ntotal} vectors, dim={dim} ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index()
