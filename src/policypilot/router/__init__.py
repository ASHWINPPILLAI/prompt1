"""
Router sub-package: question classification into LOOKUP / CALC / OUT_OF_SCOPE / ADVERSARIAL.

The router is trained locally (no LLM calls) using MiniLM embeddings + logistic regression.
See scripts/router_cv.py for the 25-run evaluation protocol.
"""
