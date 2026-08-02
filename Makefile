.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip
FI := .venv/bin/football-insights
TEST_MATCH ?= Sample_Game_2

.PHONY: help setup data prepare train evaluate export benchmark drift serve demo demo-build \
        demo-model test test-fast slice0 lint typecheck pyright format format-check audit check \
        codehealth clean reference container container-build container-smoke

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
	# `npm ci`, not `npm install`: the lockfile is committed, and installing
	# without it lets a local build drift inside the `^` ranges relative to
	# what CI and the container build resolve.
	cd demo && npm ci --silent && npm run build

demo: demo-build serve ## Build the demo, then serve it

demo-model: ## Train and export the synthetic-data model the public demo serves
	$(FI) demo-model

slice0: ## Run the Slice 0 vertical-path acceptance test
	$(PY) -m pytest tests/e2e/test_slice0.py -q

test: ## Run the full test suite
	$(PY) -m pytest -q

test-fast: ## Run everything except tests needing the downloaded dataset
	$(PY) -m pytest -q -m "not requires_data"

lint: ## Lint with ruff
	$(PY) -m ruff check src tests

format-check: ## Verify formatting without changing anything
	$(PY) -m ruff format --check src tests

format: ## Auto-format and fix with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Static type check with mypy
	$(PY) -m mypy

pyright: ## Strict static type check with pyright
	$(PY) -m pyright

audit: ## Check dependencies for known vulnerabilities
	# --skip-editable: this project is installed editable and is not published on
	# PyPI, so auditing it only ever reports "not found" — nothing about our
	# actual exposure. --strict is deliberately absent: it fails the run when
	# dependency *collection* fails, and skipping the editable install counts as
	# exactly that, so the two cannot be combined. Findings still fail the target
	# (pip-audit exits 1 on any advisory), and the exit status is not swallowed.
	$(PY) -m pip_audit --skip-editable

check: lint format-check typecheck pyright test ## Lint, type check and test

# ---------------------------------------------------------------- container
IMAGE ?= football-insights:deployment-test

container-build: ## Build the deployable image (frontend, model and service)
	docker build -t $(IMAGE) .

container-smoke: ## Start the built image and drive the live service over HTTP
	IMAGE=$(IMAGE) ./scripts/container-smoke.sh

container: container-build container-smoke ## Build the image, then smoke-test it

codehealth: ## CodeScene code-health delta (needs the cs CLI and a PAT)
	./scripts/codescene.sh

clean: ## Remove caches and build artifacts (leaves data/ and artifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find src tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
