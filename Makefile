.PHONY: dev test compare adversarial eval data lint typecheck clean

# --- Development & Data ---

dev:
	pip install -e ".[dev]"

data:
	python scripts/prepare_eval_data.py
	python scripts/prepare_adversarial_data.py
	python scripts/build_typological_set.py
	python scripts/validate_typological_set.py

# --- Testing ---

test:
	python scripts/validate_typological_set.py
	python -m pytest tests/ -v

lint:
	ruff check app/ eval/ tests/
	ruff format --check app/ eval/ tests/

typecheck:
	mypy app/gate/ app/graph/

# --- Evaluation ---

compare:
	python -m eval.run_comparison

adversarial:
	python -m eval.run_adversarial

eval: compare adversarial

# --- Checkpoint ---

repair-checkpoint:
	python scripts/repair_checkpoint.py --checkpoint-dir $(CHECKPOINT_DIR) --base-model $(BASE_MODEL)

fetch-checkpoint:
	python scripts/fetch_checkpoint.py

validate-checkpoint:
	python scripts/fetch_checkpoint.py --validate-only

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
