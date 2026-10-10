"""
Full router training and cross-validation implementation using scikit-learn.

Model: Logistic Regression (C=1.0, max_iter=1000, class_weight='balanced')
Inputs: MiniLM embeddings
Outputs: 4 classes (LOOKUP, CALC, OUT_OF_SCOPE, ADVERSARIAL)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROUTER_DATA_PATH = Path("router/data/router.jsonl")


def load_router_data() -> Tuple[list[str], list[str]]:
    """Load text and labels from router.jsonl."""
    if not _ROUTER_DATA_PATH.exists():
        raise FileNotFoundError(f"Router data not found: {_ROUTER_DATA_PATH}")
        
    texts, labels = [], []
    with _ROUTER_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(row["question"])
            labels.append(row["label"])
    return texts, labels


def train_router(X: np.ndarray, y: list[str]):
    """Train a logistic regression classifier on embeddings."""
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    except ImportError:
        logger.error("scikit-learn is required for router training. Please pip install scikit-learn.")
        raise
        
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        multi_class="multinomial"
    )
    model.fit(X, y)
    return model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using the shared MiniLM model."""
    from policypilot.retrieval import _get_default_embedder  # noqa: PLC0415
    embedder = _get_default_embedder()
    logger.info(f"Embedding {len(texts)} texts...")
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def evaluate_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 25):
    """Run 25-run cross-validation (StratifiedShuffleSplit)."""
    try:
        from sklearn.model_selection import StratifiedShuffleSplit  # noqa: PLC0415
        from sklearn.metrics import classification_report, accuracy_score  # noqa: PLC0415
    except ImportError:
        logger.error("scikit-learn required.")
        raise
        
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    
    accuracies = []
    
    logger.info(f"Running {n_splits}-split CV...")
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = train_router(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    logger.info(f"CV Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
    
    # Train on full set and print report for diagnostic
    final_model = train_router(X, y)
    full_preds = final_model.predict(X)
    logger.info("\n" + classification_report(y, full_preds))
    
    return mean_acc, std_acc
