# Contributing

## Getting set up

```bash
make setup
make test      # full suite, no network or dataset needed
make check     # lint + both type checkers + tests
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
changes the schema hash, which correctly invalidates existing artifacts: the service refuses to
load a model whose hash disagrees with the running code. Retrain and re-register.

If a feature's *meaning* changes without its name changing, bump `FEATURE_SCHEMA_REVISION`, which
the hash alone cannot detect.

## Style

Ruff for formatting and linting. Docstrings explain *why* where the reason is not obvious from the
code; the interesting comments in this codebase are the ones recording a decision or a measurement,
and those are worth keeping accurate.

## Type checking

Both mypy and Pyright run in strict mode, and both must pass. They disagree often enough to be worth
running together: mypy is stricter about untyped definitions, Pyright about partially-unknown types
leaking out of third-party libraries.

```bash
make typecheck   # mypy --strict
make pyright     # pyright, typeCheckingMode = strict
```

Pyright is configured in `pyrightconfig.json`, covering `src` and `tests`. Two settings there are
load-bearing and should not be removed: `venvPath`/`venv` (without them imports resolve against the
wrong interpreter) and `pythonVersion = "3.12"` (numpy's stubs use syntax that is not parsed at
3.11). Both carry the reasoning inline.

Rules for keeping it strict:

- **No file-wide suppressions**, and no excluding a source directory to make it pass.
- **`Any` belongs at boundaries only**, meaning JSON, YAML, environment variables and third-party
  results. Parse and validate into concrete types at the edge; everything inland should be
  precisely typed.
- **Prefer a typed adapter to a repeated ignore.** Where a third-party library is under-annotated,
  wrap it once (see `models/_torch_typing.py` and `tests/support.py`) rather than suppressing at
  every call site.
- **Every retained ignore names its diagnostic and its reason**, e.g.
  `# pyright: ignore[reportUnknownMemberType]` with a comment saying which upstream signature is at
  fault. A bare `# type: ignore` will not pass review.
- For a library with no `py.typed`, add a partial stub under `stubs/` covering only what is used.
  See `stubs/README.md`.

## Code health

CodeScene is used as a supplementary check. It needs a personal access token, so it is deliberately
*not* part of `make check`: no contributor should need proprietary credentials to submit a change.

```bash
make codehealth   # delta against the merge base with main
```

Findings are evaluated, not blindly satisfied. Vectorised numerical code, the feature builder and
the label definition carry genuine domain complexity, and a finding against them is often worth
retaining with a note in the pull request rather than refactoring away. Do not relax a CodeScene
rule to improve a score.
