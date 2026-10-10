.PHONY: setup check data ingest index eval-dev eval-test sweep \
        router-cv router-final token-budget report app clean

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Quality ───────────────────────────────────────────────────────────────────
check:
	ruff check . && pytest -q

# ── Data pipeline (run in order) ─────────────────────────────────────────────
data:
	python scripts/data/download_all.py
	python scripts/data/build_eval_sets.py
	python scripts/data/build_router_data.py

# ── Ingestion & indexing ──────────────────────────────────────────────────────
ingest:
	python -m policypilot.ingest

index:
	python -m policypilot.retrieval.build

# ── Evaluation ────────────────────────────────────────────────────────────────
eval-dev:
	python scripts/run_eval.py --split dev --versions all

eval-test:
	python scripts/run_final_eval.py

# ── Parameter sweeps ─────────────────────────────────────────────────────────
sweep:
	python scripts/run_sweep.py

# ── Token budget table ────────────────────────────────────────────────────────
token-budget:
	python scripts/token_budget.py

# ── Router ────────────────────────────────────────────────────────────────────
router-cv:
	python scripts/router_cv.py

router-final:
	python scripts/router_final.py

# ── Report assets ─────────────────────────────────────────────────────────────
report:
	python scripts/make_report_assets.py

# ── App ───────────────────────────────────────────────────────────────────────
app:
	streamlit run src/policypilot/app/main.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
