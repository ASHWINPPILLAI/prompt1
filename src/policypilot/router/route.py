"""
Router prediction interface.

Loads the trained router model from models/router.joblib and predicts
the class of a question along with its confidence score.

Classes: LOOKUP, CALC, OUT_OF_SCOPE, ADVERSARIAL
Returns FALLBACK if max probability < threshold (configs/default.yaml: router.threshold).

WHY a trained classifier rather than an LLM for routing?
- Zero cost per call (no API needed)
- Deterministic results
- Fast (<5ms per prediction)
- The 97% accuracy target is meaningful precisely because it's a trained model

WHY logistic regression on MiniLM embeddings?
- Small data (≤500 per class), fast, interpretable, and strong when embeddings
  already capture semantic meaning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/router.joblib")
_DEFAULT_THRESHOLD = 0.60
_FALLBACK_LABEL = "FALLBACK"

_model = None  # Loaded lazily
_embedder = None  # Shared with retrieval module


def _load_model():
    """Load the trained router model lazily."""
    global _model
    if _model is not None:
        return _model
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Router model not found at {_MODEL_PATH}. "
            "Run 'make router-final' (scripts/router_final.py) first."
        )
    import joblib  # noqa: PLC0415
    _model = joblib.load(_MODEL_PATH)
    logger.info("Router model loaded from %s", _MODEL_PATH)
    return _model


def _get_embedder():
    """Shared MiniLM embedder (same as retrieval, avoid loading twice)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    from policypilot.retrieval import _get_default_embedder  # noqa: PLC0415
    _embedder = _get_default_embedder()
    return _embedder


def predict(question: str, threshold: float | None = None) -> Tuple[str, float]:
    """
    Predict the routing class for a question.

    Returns:
        (label, confidence) where label is one of:
        LOOKUP, CALC, OUT_OF_SCOPE, ADVERSARIAL, or FALLBACK
        and confidence is the max class probability.
    """
    if threshold is None:
        # Load from config
        try:
            import yaml  # noqa: PLC0415
            cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
            threshold = cfg.get("router", {}).get("threshold", _DEFAULT_THRESHOLD)
        except Exception:
            threshold = _DEFAULT_THRESHOLD

    model = _load_model()
    embedder = _get_embedder()

    vec = embedder.encode([question], normalize_embeddings=True)
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    max_prob = float(probs.max())
    label = classes[probs.argmax()]

    if max_prob < threshold:
        logger.debug("Router confidence %.2f < %.2f → FALLBACK", max_prob, threshold)
        return _FALLBACK_LABEL, max_prob

    logger.debug("Router: %s (confidence=%.2f)", label, max_prob)
    return str(label), max_prob
