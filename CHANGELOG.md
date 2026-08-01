# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-01

First working system, end to end.

### Added
- Ingestion for both Metrica formats: CSV (games 1–2) and EPTS/FIFA XML + JSON (game 3), including
  the eleven `DataFormatSpecification` layouts game 3 uses to re-map columns at substitutions.
- Validation that fails loudly on unordered frames, backwards timestamps, implausible coordinates
  and unalignable events, and reports non-fatal data-quality findings.
- Attacking-direction inference from four tiers of evidence with an audit artifact, structural
  checks, and a documented override path.
- `CausalEventView`: forward-blind event access that hides in-flight event outcomes by construction.
- Penalty-area-entry labelling with configurable window, horizon and stride, plus episode grouping.
- 39 identity-invariant features with a versioned schema hash the serving path enforces.
- Baselines (logistic regression, gradient boosting), a compact GRU, and a labelled rule-based
  fallback measured separately.
- Evaluation at window and episode level with cluster-bootstrap intervals over possession sequences.
- ONNX export with standardisation folded into the graph, parity checking and latency benchmarking.
- Deterministic replay fault injection (`clean`, `jitter`, `degraded`, `hostile`).
- FastAPI service with health, readiness, model metadata, prediction, SSE stream and Prometheus
  metrics in two namespaces; React + canvas demo.
- Drift and data-quality reporting.
- 115 tests, none requiring network or the dataset.

### Known limitations
- Three matches, 196 episodes. Wide fold-to-fold spread; no interval captures between-match variance.
- Median warning time ~0.6 s at the chosen operating point.
- The GRU is the least well-calibrated of the four models; usable for ranking, not as a probability.
- Bootstrap recall intervals are biased low under one-to-one alarm matching.
- `docker build` has not been executed locally (no daemon on the development machine).
