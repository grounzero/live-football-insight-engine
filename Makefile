.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip
FI := .venv/bin/football-insights
TEST_MATCH ?= Sample_Game_2

.PHONY: help setup data prepare train evaluate export benchmark drift serve demo demo-build \
        test test-fast slice0 lint typecheck format audit check clean reference

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install the package with dev extras
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@$(PY) -c "import torch, onnxruntime, sklearn; print('torch', torch.__version__, '| onnxruntime', onnxruntime.__version__, '| sklearn', sklearn.__version__)"

data: ## Download the Metrica sample data (~180 MB, never committed)
	$(FI) acquire

prepare: ## Parse, validate, orient, feature-ise and label every match
	$(FI) prepare

train: ## Train the reference models and register artifacts
	$(FI) train --test-match $(TEST_MATCH)

evaluate: ## Leave-one-match-out cross-validation with bootstrap intervals
	$(FI) evaluate

export: ## Export the temporal model to ONNX and check parity
	$(FI) export

benchmark: ## Benchmark PyTorch against ONNX Runtime
	$(FI) benchmark

drift: ## Data-quality and distribution drift report
	$(FI) drift

reference: data prepare train evaluate export benchmark ## Reproduce the full reference run

serve: ## Start the API and demo on http://127.0.0.1:8000
	$(FI) serve --match $(TEST_MATCH) --speed 8

demo-build: ## Build the React demo into the package's static directory
	cd demo && npm install --silent && npm run build

demo: demo-build serve ## Build the demo, then serve it

slice0: ## Run the Slice 0 vertical-path acceptance test
	$(PY) -m pytest tests/e2e/test_slice0.py -q

test: ## Run the full test suite
	$(PY) -m pytest -q

test-fast: ## Run everything except tests needing the downloaded dataset
	$(PY) -m pytest -q -m "not requires_data"

lint: ## Lint with ruff
	$(PY) -m ruff check src tests

format: ## Auto-format and fix with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Static type check with mypy
	$(PY) -m mypy

audit: ## Check dependencies for known vulnerabilities
	$(PY) -m pip_audit --strict --ignore-vuln GHSA-none || true

check: lint typecheck test ## Lint, type check and test

clean: ## Remove caches and build artifacts (leaves data/ and artifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find src tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
