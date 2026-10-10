"""
Retrieval module — FAISS-backed semantic search over Doc2Dial chunks.

Build the index once with `make index` (retrieval/build.py).
Query at runtime with retrieve().

WHY MiniLM? Free, local, no rate limits, and the same embedder is reused for
the router features — one embedding model for the whole system.
WHY FAISS flat index? Exact nearest-neighbour search, easy to explain in the
report, and 396 documents fit comfortably in RAM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_DIR = Path("data/index")
_CHUNKS_PATH = Path("data/processed/chunks.jsonl")


class Retriever:
    """
    Semantic retriever backed by a FAISS flat inner-product index.

    Vectors are L2-normalised before indexing, so inner product == cosine similarity.
    """

    def __init__(
        self,
        index_dir: Path = _INDEX_DIR,
        chunks_path: Path = _CHUNKS_PATH,
    ) -> None:
        self._index_dir = index_dir
        self._chunks: list[dict[str, Any]] = []
        self._index = None          # faiss.IndexFlatIP, loaded lazily
        self._loaded = False
        self._chunks_path = chunks_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import faiss  # noqa: PLC0415 (lazy import — avoid hard dep at import time)

        index_path = self._index_dir / "chunks.index"
        meta_path = self._index_dir / "chunks_meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_dir}. "
                "Run 'make index' (python -m policypilot.retrieval.build) first."
            )

        self._index = faiss.read_index(str(index_path))
        self._chunks = [
            json.loads(line)
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info("Loaded FAISS index: %d vectors", self._index.ntotal)
        self._loaded = True

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        embedder=None,  # sentence_transformers.SentenceTransformer, injected
    ) -> list[dict[str, Any]]:
        """
        Return top-k chunks for a query.

        Each result: {clause_id, doc_id, domain, text, score, section_id}
        """
        self._ensure_loaded()
        if embedder is None:
            embedder = _get_default_embedder()

        vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results


# ── Shared embedder singleton (lazy) ─────────────────────────────────────────
_embedder = None


def _get_default_embedder():
    """Load MiniLM once; reuse for all queries."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 …")
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedder ready")
    return _embedder


# ── Module-level default retriever ────────────────────────────────────────────
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Convenience function: retrieve top-k chunks for a query."""
    return get_retriever().retrieve(query, k=k, embedder=_get_default_embedder())
