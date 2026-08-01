# Contributing

## Getting set up

```bash
make setup
make test      # 115 tests, no network or dataset needed
make check     # lint + type check + tests
```

Python 3.11 or newer; developed on 3.14.

## Before opening a pull request

`make check` must pass. Beyond that, three things carry more weight here than elsewhere:

**Do not weaken the causality guarantees.** Features are built through `CausalEventView`, which
cannot see the future. If a change needs information the view will not give it, that is the design
working. Adding a bypass silently invalidates every reported metric.

**Do not tune on held-out data.** Thresholds and episode-grouping parameters are chosen on training
matches and frozen. If you change how they are selected, say so in the PR and re-run
`make evaluate` so the numbers move together.

**Report what you measured.** If a change makes results worse, put the worse numbers in the PR. The
README states plainly that the GRU does not beat the baselines on PR-AUC; that kind of statement is
the point of the project, not an embarrassment.

## Adding a feature to the model

Feature order is part of the contract. Adding, removing or reordering entries in `FEATURE_NAMES`
changes the schema hash, which correctly invalidates existing artifacts — the service refuses to
load a model whose hash disagrees with the running code. Retrain and re-register.

If a feature's *meaning* changes without its name changing, bump `FEATURE_SCHEMA_REVISION`, which
the hash alone cannot detect.

## Style

Ruff for formatting and linting, mypy in strict mode. Docstrings explain *why* where the reason is
not obvious from the code; the interesting comments in this codebase are the ones recording a
decision or a measurement, and those are worth keeping accurate.
